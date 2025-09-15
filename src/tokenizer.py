from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import json
import numpy as np
import re
import argparse
import os
import time
import onnxruntime as ort
from transformers import AutoTokenizer, PretrainedConfig, GenerationConfig
from huggingface_hub import snapshot_download
import torch
import librosa
from FastAudioSR import FASR
from decoder import AudioTokenizer

class TTSCodec:
    def __init__(self, wav2vec2_path="facebook/wav2vec2-large-xlsr-53", tokenizer_path="YaTharThShaRma999/pretrained_tts_tokenizers", device='cuda:0'):
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            wav2vec2_path
        )
        self.feature_extractor = Wav2Vec2Model.from_pretrained(
            wav2vec2_path,
            torch_dtype=torch.float16,
        ).to(device)
      
        self.feature_extractor.config.output_hidden_states = True
        sess_options = ort.SessionOptions()
        providers = [
            ("CUDAExecutionProvider", {"device_id": 0})
        ]
        decoder_paths = snapshot_download(tokenizer_path, force_download=True)
        self.m_spectro = ort.InferenceSession(f"{decoder_paths}/m_spectro.onnx", sess_options, providers=providers)
        self.s_encoder = ort.InferenceSession(f"{decoder_paths}/s_encoder.onnx", sess_options, providers=providers)
        self.q_encoder = ort.InferenceSession(f"{decoder_paths}/q_encoder.onnx", sess_options, providers=providers)
        #self.vocoder = ort.InferenceSession(f"{decoder_paths}/b_decoder_new.onnx", sess_options, providers=providers)
        
        self.upsampler = FASR(f'{decoder_paths}/upsampler.pth')
        self.upsampler.model.half().eval()
        self.processor_tokenizer = ort.InferenceSession(f"{decoder_paths}/processer.onnx", sess_options, providers=providers)
        self.audio_detokenizer = AudioTokenizer(f'{decoder_paths}/detokenizer.safetensors')

        self.hidden_state_layer = 10
    def get_ref_clip(self, wav: np.ndarray) -> np.ndarray:

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

        features = self.feature_extractor(inputs.to(self.feature_extractor.device).half())

        features = features.hidden_states[self.hidden_state_layer].float()
        return features
    def wav2token(self, wav, duration=5):
        
        audio, sr = librosa.load(wav, sr=16000, duration=duration)
      
        ref_clip = self.get_ref_clip(audio)
        wav_ref = torch.from_numpy(ref_clip).unsqueeze(0).float()
      
        feat = self.extract_wav2vec2_features(audio)
        speech_tokens = self.q_encoder.run(["semantic_tokens"], {"features": feat.cpu().detach().numpy()})

        mel = self.m_spectro.run(["mel_spectrogram"], {"raw_waveform_with_channel": wav_ref.unsqueeze(0).cpu().numpy()}) 
        new_arr = np.transpose(mel[0], (0, 2, 1))
        context_tokens = self.s_encoder.run(["global_tokens"], {"mel_spectrogram": new_arr}) 
        return context_tokens, speech_tokens
      
    def token2wav(self, context_tokens, speech_tokens, llm_generated=False, upsample=True):
        if llm_generated:
            speech_tokens = self.extract_speech_tokens(speech_tokens)
        lowres_wav = self.detokenize(context_tokens, speech_tokens)
        if upsample:
            lowres_wav = lowres_wav.squeeze(1).half()
            wav = self.upsampler.run(lowres_wav)
        return wav
        
    def detokenize(self, context_tokens, speech_tokens):
        x = self.processor_tokenizer.run(["preprocessed_output"], {"context_tokens": context_tokens, "speech_tokens": speech_tokens})
        x = torch.from_numpy(x[0]).to("cuda:0").half()
        lowres_wav = self.audio_detokenizer.decode(x)
        return lowres_wav

        
    def format_prompt(self, text_prompt, context_tokens):
        context_tokens = "".join(
            [f"<|context_token_{i}|>" for i in context_tokens.squeeze()]
        )
        prompt = f"<|task_tts|><|start_text|>{text_prompt}<|end_text|><|context_audio_start|>{context_tokens}<|context_audio_end|>"
        return prompt
        
    def extract_speech_tokens(self, generated_output):
        pred_semantic_ids = (
            torch.tensor([int(token) for token in re.findall(r"speech_token_(\d+)", generated_output)])
            .long()
            .unsqueeze(0)
        ).numpy()
        return pred_semantic_ids   
