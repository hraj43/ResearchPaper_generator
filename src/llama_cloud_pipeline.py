import os
from llama_cloud.client import LlamaCloud
from llama_cloud.types import CloudDocumentCreate
from llama_index.llms.openai import OpenAI
from llama_index.core.async_utils import run_jobs
from llama_index.core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List

class Metadata(BaseModel):
    author_names: List[str] = Field(default_factory=list, description="List of author names")
    author_companies: List[str] = Field(default_factory=list, description="List of author companies")
    ai_tags: List[str] = Field(default_factory=list, description="AI-related tags")

def create_llamacloud_pipeline(pipeline_name, embedding_config, transform_config, data_sink_id=None):
    """
    Create a pipeline in LlamaCloud
    
    Args:
        pipeline_name (str): Name of the pipeline
        embedding_config (dict): Embedding configuration
        transform_config (dict): Transformation configuration
        data_sink_id (str, optional): Data sink ID. Defaults to None.
    
    Returns:
        tuple: LlamaCloud client and created pipeline
    """
    client = LlamaCloud(token=os.getenv("LLAMA_CLOUD_API_KEY"))

    pipeline = {
        'name': pipeline_name,
        'transform_config': transform_config,
        'embedding_config': embedding_config,
        'data_sink_id': data_sink_id
    }

    pipeline = client.pipelines.upsert_pipeline(request=pipeline)

    return client, pipeline


async def get_papers_metadata(llm, text):
    """
    Extract metadata from research paper text
    
    Args:
        llm (OpenAI): Language model instance
        text (str): Paper text
    
    Returns:
        Metadata: Extracted metadata
    """
    prompt = f"""Generate authors names, authors companies, and general top 3 AI tags for the given research paper.

    Research Paper:
    {text}

    Respond with a JSON that includes:
    - author_names: List of author names
    - author_companies: List of author companies
    - ai_tags: List of 3 AI-related tags"""

    response = await llm.acomplete(prompt)
    
    # You might need to parse the response manually
    try:
        import json
        parsed_response = json.loads(response.text)
        return Metadata(**parsed_response)
    except Exception as e:
        print(f"Error parsing metadata: {e}")
        return Metadata()
async def get_document_upload(document, llm):
    """
    Prepare document for cloud upload
    
    Args:
        document (list): Parsed document
        llm (OpenAI): Language model instance
    
    Returns:
        CloudDocumentCreate: Prepared document for upload
    """
    text_for_metadata_extraction = document[0].text + document[1].text + document[2].text
    full_text = "\n\n".join([doc.text for doc in document])
    metadata = await get_papers_metadata(llm, text_for_metadata_extraction)
    return CloudDocumentCreate(
        text=full_text,
        metadata={
            'author_names': metadata.author_names,
            'author_companies': metadata.author_companies,
            'ai_tags': metadata.ai_tags
        }
    )

async def upload_documents(client, pipeline, documents, llm):
    """
    Upload documents to LlamaCloud
    
    Args:
        client (LlamaCloud): LlamaCloud client
        pipeline (Pipeline): Created pipeline
        documents (list): Documents to upload
        llm (OpenAI): Language model instance
    """
    extract_jobs = []
    for document in documents:
        extract_jobs.append(get_document_upload(document, llm))

    document_upload_objs = await run_jobs(extract_jobs, workers=4)

    _ = client.pipelines.create_batch_pipeline_documents(pipeline.id, request=document_upload_objs)