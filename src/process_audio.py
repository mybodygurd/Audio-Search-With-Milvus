import torch
import torchaudio
from io import BytesIO

def get_file_extension(file_name: str) -> str | None:
    file_formats = {"wav", "mp3", "m4a"}
    ext = file_name.split(".")[-1].lower()
    if ext in file_formats:
        return ext
    return None

def load_and_resample_audio(file: BytesIO, target_sr: int, audio_format: str):
    waveform, sr = torchaudio.load(file, format=audio_format, backend="ffmpeg")
    
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=target_sr)
    
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    return waveform

def split_audio(waveform, chunk_duration_sec: int, target_sr: int) -> list:
    chunk_samples = int(target_sr * chunk_duration_sec)
    total_samples = waveform.shape[1]

    chunks = []
    for start in range(0, total_samples, chunk_samples):
        end = start + chunk_samples
        chunk = waveform[:, start:end]
        if chunk.shape[1] > 0.5 * chunk_samples: 
            chunks.append(chunk)

    return chunks