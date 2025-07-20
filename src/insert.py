import torchaudio
import uuid
from milvus_utils import get_milvus_client, create_vector_db
from minio_utils import get_minio_client, upload_bytes
from process_audio import load_and_resample_audio, split_audio, get_file_extension
from encoder import FeatureExtractor
from config import *
from io import BytesIO
from tqdm import tqdm



def insert(files):
    audio_encoder = FeatureExtractor(MODEL_NAME)
    milvus_client = get_milvus_client(uri=MILVUS_ENDPOINT)
    minio_client = get_minio_client(
        end_point=MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY
    )
    create_vector_db(
        client=milvus_client, collection_name=COLLECTION_NAME, dim=int(MODEL_DIM)
    )

    data = []
    for file in tqdm(files):
        try:
            obj_file = BytesIO(file.read())
            obj_file.seek(0)
            ext = get_file_extension(file.name)
            if ext is None:
                raise FileExistsError(f"couldn't extact the extention of file: {file.name}")
            waveform = load_and_resample_audio(
                file=obj_file,
                target_sr=TARGET_SAMPLE_RATE,
                audio_format=ext
            )
            chunks = split_audio(
                waveform=waveform,
                chunk_duration_sec=CHUNK_DURATION_RATE,
                target_sr=TARGET_SAMPLE_RATE
            )
            for idx, chunk in enumerate(chunks):
                buff = BytesIO()
                torchaudio.save(
                    uri=buff,
                    src=chunk,
                    sample_rate=16000,
                    format="wav"
                )
                key = f"{file.name}/{uuid.uuid4()}_chunk_{idx}.wav"
                audio_url = upload_bytes(
                    client=minio_client,
                    buffer=buff,
                    object_key=key,
                    endpoint=MINIO_ENDPOINT,
                    bucket_name=MINIO_BUCKET_NAME
                )

                emb = audio_encoder(chunk)
                data.append(
                    {
                        "vector": emb.tolist(),
                        "object_key": key,
                        "start": int(idx * CHUNK_DURATION_RATE)
                    }
                )
        except Exception as e:
            print(f"Skipping file: {file.name} due to an error occurs during the embedding process:\n{e}")
            continue

    milvus_res = milvus_client.insert(
        collection_name=COLLECTION_NAME,
        data=data
    )
    print("Total number of inserted entities/images:", milvus_res["insert_count"])
    

