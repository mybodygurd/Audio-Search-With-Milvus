import streamlit as st
from io import BytesIO
from insert import insert
from process_audio import load_and_resample_audio, get_file_extension
from encoder import FeatureExtractor
from milvus_utils import semantic_search, get_milvus_client
from minio_utils import download_audio, get_minio_client
from config import *

milvus_client = get_milvus_client(uri=MILVUS_ENDPOINT)
minio_client = get_minio_client(
        end_point=MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY
    )

@st.cache_resource
def get_audio_encoder():
    return FeatureExtractor(MODEL_NAME)

@st.cache_data
def get_audio_embedding(file, _audio_encoder):
    file_bytes = BytesIO(file.read())
    file_bytes.seek(0)
    ext = get_file_extension(query_file.name)
    if ext is None:
        raise FileExistsError(f"couldn't extract the extension file: {query_file.name}")
    resampled_audio = load_and_resample_audio(
        file=file_bytes,
        target_sr=TARGET_SAMPLE_RATE,
        audio_format=ext
    )   
    return _audio_encoder(resampled_audio)


st.title("Audio Search Wtih Milvus")
st.markdown(
    """
    Upload your audio files to index them in Milvus, then search by uploading a query audio file. 
    The system will find the most similar segments among your data.
    """
)

cols = st.columns([1, 1, 1])

with st.sidebar:
    st.image("./pics/Milvus_Logo_Official.png", width=200)
    uploaded_files = st.file_uploader(
        "Upload audio files...",
        type=["wav", "mp3", "m4a", "ogg", "aac"],
        accept_multiple_files=True
    )
    
if uploaded_files:
    if st.sidebar.button("Index Audio Files"):
        st.write("Indexing audio files to Milvus & Minio...")
        insert(
            files=uploaded_files,
            encoder=get_audio_encoder(),
            milvus_client=milvus_client,
            minio_client=minio_client
        )
        st.success("Files indexed successfully.")
    query_file = st.sidebar.file_uploader(
        "Upload audio file...",
        type=["wav", "mp3", "m4a", "ogg", "aac"]
    )
    if query_file:
        audio_encoder = get_audio_encoder()
        st.info("Performing Semantic Search...")
        results = semantic_search(
            client=milvus_client,
            collection_name=COLLECTION_NAME,
            query_vector=get_audio_embedding(query_file, audio_encoder)
        )
        search_results = results[0]
        if not search_results:
            st.warning("No similar audio segments found.")
        else:
            for i, info in enumerate(search_results):
                object_key = info["entity"]["object_key"]
                start = info["entity"]["start"]
                score = info["distance"]
                audio_bytes = download_audio(
                    client=minio_client,
                    bucket_name=MINIO_BUCKET_NAME,
                    object_name=object_key
                )
                with cols[i % 3]:
                    st.audio(audio_bytes, format="audio/wav")
                    st.write(f"Start from: {start}")
                    st.write(f"Score: {score:.4f}")
                    st.write(f"File: {object_key}")
                    
    
