import os
import pandas as pd

from towhee import pipe, ops
from towhee.datacollection import DataCollection
from pymilvus import MilvusClient, DataType
from sklearn.metrics import accuracy_score


COLLECTION_NAME = "nnfp"
DATABASE_NAME = "milvus_demo.db"
HOST = "localhost"
PORT = "19530"
DIM = 128

def embedding_audio(path):
    emb_pipe = (
        pipe.input("path")
            .map("path", "frames", ops.audio_decode.ffmpeg())
            .flat_map("frames", "fingerprints", ops.audio_embedding.nnfp(device="cpu"))
            .map(("fingerprints", "path"), "milvus_res", ops.ann_insert.milvus(
                host=HOST,
                port=PORT,
                collection_name=COLLECTION_NAME,
                dim=DIM,
                id_field="id",
                insert_field="vector",
                extra_field="audio"
            ))
            .output()
    )
    print("="*100)
    print(DataCollection(emb_pipe(path)).show())




def main():
    df = pd.read_csv("audio_fp/ground_truth.csv")

    path = df["answer"][0]
    client = create_vector_db()
    embedding_audio(path)





# def insert_data_to_db(client, df: pd.DataFrame, vectors):
#    data = [
#        {"id": i, "vector": vectors[i], "audio": df["answer"][i]}
#        for i in range(len(vectors))
#    ]
#    print("Data has", len(data), "entities, each with fields: ", data[0].keys())
#    print("Vector dim:", len(data[0]["vector"]))
#    print('-' * 100)
#
#    res = client.insert(collection_name=COLLECTION_NAME, data=data)
#    print(res)


if __name__ == "__main__":
    main()