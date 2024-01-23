import numpy as np

from llama_cpp.llama_speculative import LlamaPromptLookupDecoding

def test_find_candidate_pred_tokens():
    find_candidate_pred_tokens = LlamaPromptLookupDecoding.find_candidate_pred_tokens

    # Test Case 1: Matching ngram is found
    input_ids1 = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
    result1 = find_candidate_pred_tokens(input_ids1, max_ngram_size=3, num_pred_tokens=2)
    assert np.array_equal(result1, np.array([1, 2]))

    # Test Case 2: Matching ngram is not found
    input_ids2 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    result2 = find_candidate_pred_tokens(input_ids2, max_ngram_size=3, num_pred_tokens=2)
    assert np.array_equal(result2, np.array([]))
