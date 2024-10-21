import modal
import pytest


@pytest.mark.parametrize(
    "model,query,expected_strs",
    [
        (
            "gpt-4o-mini",
            "What were the results of the Needle In A Haystack evaluation? And what type of cluster and GPU was used for experiments?",
            ["NVLink", "GPU", "NVIDIA", "H800"],
        ),
        (
            "gpt-4o-mini",
            "What is the conclusion of this paper?",
            [
                "DeepSeek-V2",
                "MoE",
            ],
        ),
    ],
)
def test_answer_questions_with_image_context(model, query, expected_strs):
    answer_questions_with_image_context = modal.Function.lookup("multi-modal-rag", "answer_questions_with_image_context")
    res = answer_questions_with_image_context.remote_gen(
        pdf_url="https://arxiv.org/pdf/2405.04434",
        queries=[query],
        top_k=5,
        use_cache=True,
        max_new_tokens=1024,
        model=model,
    )
    chunks = [r for r in res]
    answer_str = "".join([c for c in chunks if isinstance(c, str)])

    all_images_data = [c for c in chunks if not isinstance(c, str)]

    assert len(all_images_data) == 1
    assert len(all_images_data[0][0]) == 5
    assert all(["data:image" in img for img in all_images_data[0][0]])
    assert all(t in answer_str for t in expected_strs)
