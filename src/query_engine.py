from llama_index.indices.managed.llama_cloud import LlamaCloudIndex

def create_query_engine(api_key, index_name="report_generation", project_name="Default"):
    """
    Create a query engine for the LlamaCloud index
    
    Args:
        api_key (str): LlamaCloud API key
        index_name (str, optional): Name of the index. Defaults to "report_generation".
        project_name (str, optional): Project name. Defaults to "Default".
    
    Returns:
        QueryEngine: Configured query engine
    """
    index = LlamaCloudIndex(
        name=index_name,
        project_name=project_name,
        api_key=api_key
    )

    query_engine = index.as_query_engine(
        dense_similarity_top_k=10,
        sparse_similarity_top_k=10,
        alpha=0.5,
        enable_reranking=True,
        rerank_top_n=5,
        retrieval_mode="chunks"
    )

    return query_engine