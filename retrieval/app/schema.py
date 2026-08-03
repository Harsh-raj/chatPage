from pydantic import BaseModel


class Document(BaseModel):
    id: str
    text: str
    metadata: dict = {}


class IndexRequest(BaseModel):
    documents: list[Document]


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResult(BaseModel):
    id: str
    text: str
    score: float
    metadata: dict = {}


class SearchResponse(BaseModel):
    results: list[SearchResult]


class DeleteByIdsRequest(BaseModel):
    ids: list[str]


class DeleteResponse(BaseModel):
    deleted: int
    remaining_documents: int


class DocumentSummary(BaseModel):
    source: str
    chunk_count: int


class ListDocumentsResponse(BaseModel):
    documents: list[DocumentSummary]
    total_chunks: int