import torch
import torchaudio
from milvus_utils import get_milvus_client, create_vector_db
from encoder import FeatureExtractor

import os
import sys
from glob import glob
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

TARGET_SAMPLE_RATE = os.getenv("TARGET_SAMPLE_RATE")
CHUNK_DURATION_RATE = os.getenv("CHUNK_DURATION_RATE")
MIVUS_ENDPOINT = os.getenv("MIVUS_ENDPOINT")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
MODEL_NAME = os.getenv("MODEL_NAME")
MODEL_DIM = os.getenv("DIM")


def load_and_resample_audio(file_path: str, target_sr: int=TARGET_SAMPLE_RATE):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file {file_path} not found.")
    waveform, sr = torchaudio.load(file_path)
    
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=target_sr)
    
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    return waveform

def split_audio(waveform, chunck_duration_sec: int=CHUNK_DURATION_RATE) -> list:
    chunk_samples = int(TARGET_SAMPLE_RATE * chunck_duration_sec)
    total_samples = waveform[1] * TARGET_SAMPLE_RATE

    chunks = []
    for start in range(0, total_samples, chunk_samples):
        end = start + chunk_samples
        chunk = waveform[:, start:end]
        if chunk.shape[1] > 0.5 * chunk_samples: 
            chunks.append(chunk)

    return chunks

def insert(milvus_client, collection_name, data):
    milvus_res = milvus_client.insert(
        collection_name=collection_name,
        data=data
    )
    return milvus_res

def main():
    data_dir = sys.argv[-1]
    audio_encoder = FeatureExtractor(MODEL_NAME)
    milvus_client = get_milvus_client(uri=MIVUS_ENDPOINT)

    create_vector_db(
        milvus_client= milvus_client, collection_name=COLLECTION_NAME, dim=int(MODEL_DIM)
    )

    audio_paths = glob(os.path.join(data_dir, "*.wav"))

    data = []
    for filepath in enumerate(tqdm(audio_paths)):
        try:
            waveform = load_and_resample_audio(filepath)
            chunks = split_audio(waveform)
            for chunk in chunks:
                emb = audio_encoder(chunk)
                data.append(
                    {
                        "embedding": emb,
                        "source_file": filepath
                    }
                )
        except Exception as e:
            print(f"Skipping file: {filepath} due to an error occurs during the embedding process:\n{e}")
            continue

    milvus_res = insert(milvus_client, COLLECTION_NAME, data)
    print("Total number of inserted entities/images:", milvus_res["insert_count"])
    





if __name__ == "__main__":
    main()
