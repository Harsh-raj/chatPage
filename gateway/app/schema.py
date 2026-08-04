from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class SourceChunk(BaseModel):
    id: str
    text: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
