from typing import List

import modal
import pytest
import torch

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
    (
        "https://arxiv.org/pdf/2004.12832",
        ["What is the general architecture of ColBERT given a query q and a document d?"],
        [2],
    ),
]


@pytest.mark.parametrize("url, questions, expected_indices", TEST_CASES)
def test_top_pages_no_cache(url: str, questions: List[str], expected_indices: List[int]):
    top_pages = modal.Function.lookup("pdf-retriever", "PDFRetriever.top_pages")

    results = top_pages.remote(url, questions, use_cache=False, top_k=1)
    actual_indices = [a[0] for a in results]
    assert actual_indices == expected_indices, f"Expected {expected_indices}, but got {actual_indices}"


@pytest.mark.parametrize("url, questions, expected_indices", TEST_CASES)
def test_top_pages_cache(url: str, questions: List[str], expected_indices: List[int]):
    top_pages = modal.Function.lookup("pdf-retriever", "PDFRetriever.top_pages")

    # Test with use_cache=True
    results = top_pages.remote(url, questions, use_cache=True, top_k=1)
    actual_indices = [a[0] for a in results]
    assert actual_indices == expected_indices, f"Expected {expected_indices}, but got {actual_indices}"


def test_top_pages_top_k():
    top_pages = modal.Function.lookup("pdf-retriever", "PDFRetriever.top_pages")
    results = top_pages.remote(
        "https://arxiv.org/pdf/2305.07759",
        ["How many parameters do the models have when trained on TinyStories dataset?", "What is the name of the dataset?"],
        use_cache=True,
        top_k=4,
    )
    assert len(results) == 2
    assert len(results[0]) == len(results[1]) == 4
    assert 3 in results[1]
    assert 2 in results[0]


def test_forward_strings():
    forward = modal.Function.lookup("pdf-retriever", "PDFRetriever.forward")
    t1 = "The quick brown fox jumps over the lazy dog. This pangram contains every letter of the English alphabet at least once. Pangrams are often used to display fonts or test equipment. In typography, they're particularly useful for displaying the characteristics of a font."
    t2 = "Hello Friend, how are you?"
    t3 = "Hello"
    t4 = "1"
    res = forward.remote([t1])
    assert [r.shape for r in res] == [torch.Size([66, 128])]

    res = forward.remote([t2])
    assert [r.shape for r in res] == [torch.Size([21, 128])]

    res = forward.remote([t3])
    assert [r.shape for r in res] == [torch.Size([15, 128])]

    res = forward.remote([t1, t2, t3, t4])
    assert [r.shape for r in res] == [torch.Size([66, 128]), torch.Size([66, 128]), torch.Size([66, 128]), torch.Size([66, 128])]
