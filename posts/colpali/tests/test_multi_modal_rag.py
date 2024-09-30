import modal


def test_answer_questions_with_image_context():
    answer_questions_with_image_context = modal.Function.lookup("multi-modal-rag", "answer_questions_with_image_context")
    res = answer_questions_with_image_context.remote(
        pdf_url="https://arxiv.org/pdf/2310.06825",
        queries=[
            "What is the paper about?",
            "How does mamba architecture compare to other LLM architectures?",
        ],
        top_k=5,
        use_cache=True,
        max_new_tokens=1024,
    )
    assert len(res) == 2
    assert all(t in res[0][0] for t in ["NVLink", "GPU", "NVIDIA", "H800"])
    assert all(t in res[1][0] for t in ["DeepSeek-V2", "model", "128K", "performance"])
