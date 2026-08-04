from schema import IndexRequest, SearchResponse, SearchRequest
from fastapi import FastAPI

from embeddings import RealEmbedder
from langchain_qdrant.qdrant import QdrantVectorStore

embedder = RealEmbedder()

store = QdrantVectorStore(vector_size=embedder.dim if hasattr(embedder, "dim") else 64)

app = FastAPI()

@app.post("/index")
def index_documents(request: IndexRequest) -> dict:
    ids = [doc.id for doc in request.documents]
    texts = [doc.text for doc in request.documents]
    payloads = [{"text": doc.text, "metadata": doc.metadata} for doc in request.documents]

    vectors = embedder.embed(texts)      # turn each doc's text into a vector
    store.add(ids, vectors, payloads)    # store id + vector + the ORIGINAL TEXT together

@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    query_vector = embedder.embed([request.query])[0]   # embed the QUESTION the same way
    raw_results = store.search(query_vector, top_k=request.top_k)