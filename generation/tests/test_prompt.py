from app.prompt import build_rag_prompt


def test_prompt_includes_all_context_chunks():
    prompt = build_rag_prompt("What is X?", ["chunk one", "chunk two"])
    assert "chunk one" in prompt
    assert "chunk two" in prompt
    assert "What is X?" in prompt


def test_prompt_handles_no_context_gracefully():
    prompt = build_rag_prompt("What is X?", [])
    assert "No relevant context was found." in prompt
