from typing import List

import modal
import pytest

TEST_CASES = [
    (
        "https://arxiv.org/pdf/1706.03762",
        [
            "Who are the authors of the paper?",
            "What is the model architecture for the transformer?",
            "What is the equation for Scaled Dot-Product Attention?",
            "What Optimizer was used for training?",
            "What was the value used for label smoothing?",
        ],
        [0, 2, 3, 6, 7],
    ),
    (
        "https://arxiv.org/pdf/2407.01449",
        [
            "What was the size of the training dataset?",
            "Can you summarize the abstract for me please?",
            "What is the main contribution of this paper?",
        ],
        [4, 0, 1],
    ),
    (
        "https://arxiv.org/pdf/2307.09288",
        ["How many jelly beans did the jar contain", "What's a good haircut that looks great on everybody"],
        [56, 58],
    ),
]


@pytest.mark.parametrize("url, questions, expected_indices", TEST_CASES)
def test_colpali(url: str, questions: List[str], expected_indices: List[int]):
    top_pages = modal.Function.lookup("colpali", "ColPaliModel.top_pages")
    results = top_pages.remote(url, questions)
    actual_indices = [a[0] for a in results]
    assert actual_indices == expected_indices, f"Expected {expected_indices}, but got {actual_indices}"
