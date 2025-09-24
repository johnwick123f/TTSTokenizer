from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import gc
import numpy as np
import re
import os
import time
import onnxruntime as ort
from transformers import AutoTokenizer, PretrainedConfig, GenerationConfig
from huggingface_hub import snapshot_download
import torch
import librosa
from FastAudioSR import FASR
from decoder import AudioTokenizer
from tokenizer_utils import *
from transformers import pipeline

class TTSCodec:
    def __init__(self, wav2vec2_path="facebook/wav2vec2-large-xlsr-53", tokenizer_path="YaTharThShaRma999/pretrained_tts_tokenizers", device='cuda:0', whisper_model="openai/whisper-small"):
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
        decoder_paths = snapshot_download(tokenizer_path)
        self.m_spectro = ort.InferenceSession(f"{decoder_paths}/m_spectro.onnx", sess_options, providers=providers)
        self.s_encoder = ort.InferenceSession(f"{decoder_paths}/s_encoder.onnx", sess_options, providers=providers)
        self.q_encoder = ort.InferenceSession(f"{decoder_paths}/q_encoder.onnx", sess_options, providers=providers)
        #self.vocoder = ort.InferenceSession(f"{decoder_paths}/b_decoder_new.onnx", sess_options, providers=providers)
        
        self.upsampler = FASR(f'{decoder_paths}/upsampler.pth')
        self.upsampler.model.half().eval()
        self.processor_tokenizer = ort.InferenceSession(f"{decoder_paths}/processer.onnx", sess_options, providers=providers)
        self.audio_detokenizer = AudioTokenizer(f'{decoder_paths}/detokenizer.safetensors')
        self.transcriber = pipeline("automatic-speech-recognition", model=whisper_model, device='cuda:0', torch_dtype=torch.bfloat16)
        
        self.ref_segment_length = 96000
        self.hidden_state_layer = 10
    def get_ref_clip(self, wav):

        """pads to ref segment lenght"""

        ref_segment_length = self.ref_segment_length
        wav_length = len(wav)

        if ref_segment_length > wav_length:
            wav = np.tile(wav, ref_segment_length // wav_length + 1)

        return wav[:ref_segment_length]

    def extract_wav2vec2_features(self, wavs: torch.Tensor) -> torch.Tensor:
        """extract wav2vec2 hidden state for semantics"""
        inputs = self.processor(
            wavs,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            output_hidden_states=True,
        ).input_values
        feat = self.feature_extractor(inputs.to(self.feature_extractor.device).half())
        avg_feat = (
            feat.hidden_states[11] + feat.hidden_states[14] + feat.hidden_states[16]
        ) / 3

        return avg_feat.float()
        
    @torch.inference_mode()   
    def wav2token(self, audio, duration=8, use_transcription=True, add_silence=16000):

        """encodes audio file into speech tokens and context tokens"""
        audio, sr = librosa.load(audio, duration=duration, sr=16000)
        if add_silence:
            audio = np.concatenate((audio, np.zeros(add_silence)))

        ref_clip = self.get_ref_clip(audio)
        wav_ref = torch.from_numpy(ref_clip).unsqueeze(0).float()
      
        feat = self.extract_wav2vec2_features(audio)
        speech_tokens = self.q_encoder.run(["semantic_tokens"], {"features": feat.cpu().detach().numpy()})

        mel = self.m_spectro.run(["mel_spectrogram"], {"raw_waveform_with_channel": wav_ref.unsqueeze(0).cpu().numpy()}) 
        new_arr = np.transpose(mel[0], (0, 2, 1))
        context_tokens = self.s_encoder.run(["global_tokens"], {"mel_spectrogram": new_arr}) 
        if use_transcription:
            transcription = self.transcriber(audio)['text'].lstrip()
            transcription = transcription if transcription.endswith('.') else transcription + '. '
            return context_tokens, speech_tokens, transcription
        else:
            return context_tokens, speech_tokens

    @torch.inference_mode()
    def token2wav(self, context_tokens, speech_tokens, llm_generated=False, upsample=True, concat=True):

        """decodes the speech tokens with context tokens for audio output, optionally upsamples to 48khz for higher quality output"""
        
        if llm_generated:
            speech_tokens = self.extract_speech_tokens(speech_tokens)
        wav = self.detokenize(context_tokens, speech_tokens)
        if upsample:
            wav = wav.squeeze(1).half()
            wav = self.upsampler.run(wav)

        if concat:
            wav = wav.flatten()
        return wav
        
        
    @torch.inference_mode()    
    def detokenize(self, context_tokens, speech_tokens):
        """helper function to detokenize"""
        x = self.processor_tokenizer.run(["preprocessed_output"], {"context_tokens": context_tokens, "speech_tokens": speech_tokens})
        x = torch.from_numpy(x[0]).to("cuda:0").half()
        lowres_wav = self.audio_detokenizer.decode(x)
        return lowres_wav

        
    def format_prompt(self, sentences, context_tokens, speech_tokens=None, transcription=None, split=True):
        """formats prompt for llm tts model"""

        formatted_prompts = []
        context_tokens_formatted = "".join(
            [f"<|context_token_{i}|>" for i in context_tokens.squeeze()]
        )
        if speech_tokens:
            speech_tokens_formatted = "".join(
                    [f"<|speech_token_{i}|>" for i in speech_tokens.squeeze()]
            )
        if split:
            sentences = split_sentences(sentences)
            
        for sentence in sentences:
            if speech_tokens:
                prompt = f"<|task_tts|><|start_text|>{transcription}{sentence}<|end_text|><|context_audio_start|>{context_tokens_formatted}<|context_audio_end|><|prompt_speech_start|>{speech_tokens_formatted}"
            else:
                prompt = f"<|task_tts|><|start_text|>{sentence}<|end_text|><|context_audio_start|>{context_tokens_formatted}<|context_audio_end|>"
            formatted_prompts.append(prompt)
        return formatted_prompts
        
    def extract_speech_tokens(self, generated_output):
        """extracts speech tokens from llm tts model output"""
        pred_semantic_ids = (
            torch.tensor([int(token) for token in re.findall(r"speech_token_(\d+)", generated_output)])
            .long()
            .unsqueeze(0)
        ).numpy()
        return pred_semantic_ids
        
    def c_cache(self):
        """clears any vram/cache, very useful"""
        gc.collect()
        torch.cuda.empty_cache()
