from pymilvus import MilvusClient, DataType


def get_milvus_client(uri) -> MilvusClient:
    return MilvusClient(uri=uri)

def create_vector_db(
    client: MilvusClient, collection_name: str, dim: int        
):
    schema = client.create_schema()
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field(field_name="object_key", datatype=DataType.VARCHAR, max_length=500)
    schema.add_field(field_name="start", datatype=DataType.INT64)

    if client.has_collection(collection_name):
        client.drop_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        metric_type="COSINE"
    )
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="FLAT",
        metric_type="COSINE",
    )
    client.create_index(
        collection_name=collection_name,
        index_params=index_params
    )
    client.load_collection(collection_name)
    

def semantic_search(
        client: MilvusClient, collection_name: str, query_vector
):
    return client.search(
        collection_name=collection_name,
        data=[query_vector],
        search_params={"metric_type": "COSINE", "params": {"ef": 100}},
        output_fields=["object_key", "start"]
    )
