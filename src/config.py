from dotenv import load_dotenv
import os

load_dotenv()

MILVUS_ENDPOINT = os.getenv("MILVUS_ENDPOINT")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

MODEL_NAME = os.getenv("MODEL_NAME")
MODEL_DIM = os.getenv("MODEL_DIM")

TARGET_SAMPLE_RATE = int(os.getenv("TARGET_SAMPLE_RATE"))
CHUNK_DURATION_RATE = int(os.getenv("CHUNK_DURATION_RATE"))

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
MINIO_BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME")