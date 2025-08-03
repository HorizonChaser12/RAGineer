from pydantic import BaseModel
from typing import Optional, Literal


# Pydantic models for API requests
class QueryRequest(BaseModel):

    query: str
    k: Optional[int] = 3
    temperature: Optional[float] = None
    analytical_mode: Optional[bool] = False
    excel_file_path: Optional[str] = None
    sheet_name: Optional[str] = None


class InitializeRequest(BaseModel):

    excel_file_path: str
    temperature: Optional[float] = 0.7
    concise_prompt: Optional[bool] = False
    index_file: Optional[str] = "Azure_Implementation/faiss_index_azure.bin"
    use_sentence_transformers: Optional[bool] = True
    use_reranker: Optional[bool] = True
    sentence_transformer_model: Optional[str] = "all-MiniLM-L6-v2"
    reranker_model: Optional[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RebuildIndexRequest(BaseModel):

    force_rebuild: Optional[bool] = False