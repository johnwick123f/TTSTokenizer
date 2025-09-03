from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import json
import numpy as np
import re
import argparse
import os
import soundfile
import soxr
import time
import onnxruntime as ort
from transformers import AutoTokenizer, PretrainedConfig, GenerationConfig
from huggingface_hub import snapshot_download
import soundfile as sf
import torch

class TTSCodec:
    def __init__(self, wav2vec2_path="facebook/wav2vec2-large-xlsr-53", tokenizer_path="YaTharThShaRma999/pretrained_tts_tokenizers", device='cuda:0'):
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            wav2vec2_path
        )
        self.feature_extractor = Wav2Vec2Model.from_pretrained(
            wav2vec2_path
        ).to(device)
      
        self.feature_extractor.config.output_hidden_states = True
        sess_options = ort.SessionOptions()
        providers = [
            ("CUDAExecutionProvider", {"device_id": 0})
        ]
        decoder_paths = snapshot_download(tokenizer_path)
        self.m_spectro = ort.InferenceSession(f"{decoder_paths}/m_spectro.onnx", sess_options, providers=providers)
        self.s_encoder = ort.InferenceSession(f"{decoder_paths}/s_encoder.onnx", sess_options, providers=providers)
        self.q_encoder = ort.InferenceSession(f"{decoder_paths}/q_encoder.onnx", sess_options, providers=providers)
        self.vocoder = ort.InferenceSession(f"{decoder_paths}/b_decoder.onnx", sess_options, providers=providers)
    def get_ref_clip(self, wav: np.ndarray) -> np.ndarray:
        """Get reference audio clip for speaker embedding."""
        ref_segment_length = 96000
        wav_length = len(wav)

        if ref_segment_length > wav_length:
            wav = np.tile(wav, ref_segment_length // wav_length + 1)

        return wav[:ref_segment_length]
    def extract_wav2vec2_features(self, wavs: torch.Tensor) -> torch.Tensor:

        inputs = self.processor(
            wavs,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            output_hidden_states=True,
        ).input_values

        features = self.feature_extractor(inputs.to(self.feature_extractor.device))

        features = features.hidden_states[10]
        return features
    def wav2token(self, wav):
        audio, sr = soundfile.read(wav)
        audio = soxr.resample(audio, sr, 16000, quality="VHQ")
      
        ref_clip = self.get_ref_clip(audio)
        wav_ref = torch.from_numpy(ref_clip).unsqueeze(0).float()
      
        feat = self.extract_wav2vec2_features(audio)
        s_tokens = self.q_encoder.run(["semantic_tokens"], {"features": feat.cpu().detach().numpy()})

        mel = self.m_spectro.run(["mel_spectrogram"], {"raw_waveform_with_channel": wav_ref.unsqueeze(0).cpu().numpy()}) 
        new_arr = np.transpose(mel[0], (0, 2, 1))
        g_tokens = self.s_encoder.run(["global_tokens"], {"mel_spectrogram": new_arr}) 
        return s_tokens, g_tokens
      
    def token2wav(self, g_tokens, s_tokens):
        wav = self.vocoder.run(["output_waveform"], {"global_tokens": g_tokens, "semantic_tokens": s_tokens})
        return wav[0]
    

