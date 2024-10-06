import modal


def test_answer_questions_with_image_context():
    answer_questions_with_image_context = modal.Function.lookup("multi-modal-rag", "answer_questions_with_image_context")
    res, all_images_data = answer_questions_with_image_context.remote(
        pdf_url="https://arxiv.org/pdf/2405.04434",
        queries=[
            "What were the results of the Needle In A Haystack evaluation? And what type of cluster and GPU was used for experiments?",
            "What is the conclusion of this paper?",
        ],
        top_k=5,
        use_cache=True,
        max_new_tokens=1024,
    )
    assert len(res) == 2
    print(res)
    assert len(all_images_data) == 2
    assert len(all_images_data[0]) == 5 == len(all_images_data[1])
    assert all(["data:image" in img for img in all_images_data[0]])
    assert all(["data:image" in img for img in all_images_data[1]])
    assert all(t in res[0][0] for t in ["NVLink", "GPU", "NVIDIA", "H800"])
    assert all(
        t in res[1][0]
        for t in [
            "DeepSeek-V2",
            "128K",
        ]
    )
