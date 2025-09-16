import numpy as np
from pydub import AudioSegment, silence
import librosa

def remove_silence_pydub(audio_data, sample_rate, silence_thresh_db=-40, min_silence_len=300):
    """Removes silence from a NumPy array audio signal and returns a new NumPy array."""
    audio_segment = AudioSegment(audio_data.tobytes(), frame_rate=sample_rate, sample_width=audio_data.dtype.itemsize, channels=1)
    chunks = silence.split_on_silence(audio_segment, min_silence_len=min_silence_len, silence_thresh=silence_thresh_db, keep_silence=10)
    processed_audio = sum(chunks).normalize(headroom=5.0)
    return np.array(processed_audio.get_array_of_samples())

def load_audio(audio_path):
    """loads audio and processes it with pydub"""
    audio, sr = librosa.load(r"C:\Users\Nitin\Downloads\tiktok_influencer.wav")
    audio = audio * 2147483647.0
    audio = audio.astype(np.int32)
    trimmed_wav = remove_silence_pydub(audio, sr)
    return trimmed_wav

def cross_fade_chunks(prev_chunk, current_chunk, fade_samples):
    """ Performs a cross-fade between two audio chunks and returns the current chunk with a smoothed beginning. """
  
    if len(prev_chunk) < fade_samples or len(current_chunk) < fade_samples:
        raise ValueError("Chunks must be at least as long as fade_samples.")

    fade_out = np.linspace(1, 0, fade_samples)
    fade_in = np.linspace(0, 1, fade_samples)

    cross_faded_overlap = (prev_chunk[-fade_samples:] * fade_out) + (current_chunk[:fade_samples] * fade_in)
    smoothed_current_chunk = np.concatenate([cross_faded_overlap, current_chunk[fade_samples:]])

    return smoothed_current_chunk

def batch_cross_fade(audio_chunks, fade_samples=1600):
    reconstructed_audio = audio_chunks[0, :-fade_samples]
    num_chunks = audio_chunks.shape[0]
    for i in range(1, num_chunks):
        # Get the cross-faded and smoothed part of the current chunk
        smoothed_chunk_part = cross_fade_chunks(audio_chunks[i - 1], audio_chunks[i], fade_samples)

        # Append the smoothed part of the current chunk to the reconstructed audio
        reconstructed_audio = np.concatenate([reconstructed_audio, smoothed_chunk_part])
    return reconstructed_audio

def split_sentences(text):
  sentences = [s for s in re.split(r'(?<=[.!?])\s*', text) if s]
  return sentences
