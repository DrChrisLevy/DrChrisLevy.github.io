import modal


def test_answer_questions_with_image_context():
    answer_questions_with_image_context = modal.Function.lookup("multi-modal-rag", "answer_questions_with_image_context")
    res = answer_questions_with_image_context.remote(
        pdf_url="https://arxiv.org/pdf/2405.04434",
        queries=[
            "What were the results of the Needle In A Haystack evaluation? And what type of cluster and GPU was used for experiments?",
            "What is the conclusion of this paper?",
        ],
        top_k=1,
        use_cache=True,
        max_new_tokens=1024,
    )
    assert len(res) == 2
    assert all(t in res[0][0] for t in ["NVLink", "GPU", "NVIDIA", "H800"])
    assert all(t in res[1][0] for t in ["DeepSeek-V2", "model", "128K", "performance"])
