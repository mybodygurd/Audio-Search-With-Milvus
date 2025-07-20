from minio import Minio
import os

def get_minio_client(
        end_point, access_key, secret_key
):
    return Minio(
        endpoint=end_point,
        access_key=access_key,
        secret_key=secret_key,
        secure=False
    )

def upload_bytes(
        client: Minio, buffer, object_key, endpoint, bucket_name, content_type="audio/wav"
):
    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)
        print("Bucket created:", bucket_name)

    buffer.seek(0, os.SEEK_END)
    length = buffer.tell()
    buffer.seek(0, os.SEEK_SET)

    client.put_object(
        bucket_name=bucket_name,
        object_name=object_key,
        data=buffer,
        length=length,
        content_type=content_type
    )
    return f"http://{endpoint}/{bucket_name}/{object_key}"

def download_audio(
        client: Minio, bucket_name: str, object_name: str, extra_query_params=None
):
    try:
        response = client.get_object(
            bucket_name=bucket_name,
            object_name=object_name,
            extra_query_params=extra_query_params
        )
        data = response.read()
        return data
    finally:
        response.close()
        response.release_conn()