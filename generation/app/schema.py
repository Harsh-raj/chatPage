from pydantic import BaseModel


class GenerateRequest(BaseModel):
    query: str
    context_chunks: list[str] = []


class GenerateResponse(BaseModel):
    answer: str
    prompt_char_length: int
    injection_flags: list[dict] = []
