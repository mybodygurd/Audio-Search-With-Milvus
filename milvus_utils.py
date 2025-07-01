from pymilvus import MilvusClient, DataType


def get_milvus_client(uri) -> MilvusClient:
    return MilvusClient(uri=uri)

def create_vector_db(
    milvus_client: MilvusClient, collection_name: str, dim: int        
):
    schema = milvus_client.create_schema()
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field(field_name="audio", datatype=DataType.VARCHAR, max_length=500)

    if milvus_client.has_collection(collection_name):
        milvus_client.drop_collection(collection_name)

    return milvus_client.create_collection(
        collection_name=collection_name,
        schema=schema
    )
    


def semantic_search(
        milvus_client: MilvusClient, collection_name: str, query_vector, output_fields
):
    search_res = milvus_client.search(
        collection_name=collection_name,
        data=[query_vector],
    
    )