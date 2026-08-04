def build_rag_prompt(query: str, context_chunks: list[str]) -> str:
    """Assembles a grounded prompt from retrieved context and the user's query."""
    if not context_chunks:
        context_block = "No relevant context was found."
    else:
        context_block = "\n\n".join(f"[{i+1}] {chunk}" for i, chunk in enumerate(context_chunks))

    return (
        "Answer the question using ONLY the context below. "
        "If the context does not contain the answer, say you don't know.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )