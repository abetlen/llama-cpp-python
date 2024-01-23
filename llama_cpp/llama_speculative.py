import abc

from typing import Any

import numpy as np
import numpy.typing as npt


class LlamaDraftModel(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self, input_ids: npt.NDArray[np.intc], /, **kwargs: Any
    ) -> npt.NDArray[np.intc]:
        raise NotImplementedError()


class LlamaPromptLookupDecoding(LlamaDraftModel):
    """Based on https://github.com/apoorvumang/prompt-lookup-decoding"""

    def __init__(self, max_ngram_size: int = 3, num_pred_tokens: int = 10):
        self.max_ngram_size = max_ngram_size
        self.num_pred_tokens = num_pred_tokens

    @staticmethod
    def find_candidate_pred_tokens(
        input_ids: npt.NDArray[np.intc],
        max_ngram_size: int = 3,
        num_pred_tokens: int = 10,
    ):
        input_length = input_ids.shape[0]

        if input_length < max_ngram_size:
            return np.array([], dtype=np.intc)

        for ngram_size in range(max_ngram_size, 0, -1):
            # Extract the last n tokens as our search ngram
            ngram = input_ids[-ngram_size:]

            # Create sliding windows of size ngram_size
            windows = np.lib.stride_tricks.sliding_window_view(input_ids, (ngram_size,))

            # Convert ngram to an array for comparison
            ngram_array = np.array(ngram)  # .reshape(1, -1)

            # Find where the windows match the ngram
            matches = np.all(windows == ngram_array, axis=1)

            # Get the indices of matches
            match_indices = np.nonzero(matches)[0]

            # Iterate through match indices to find a valid continuation
            for idx in match_indices:
                start_idx = idx + ngram_size
                end_idx = start_idx + num_pred_tokens
                # Ensure we don't go beyond the length of input_ids and avoid self-match
                if end_idx <= input_length and start_idx < input_length - ngram_size:
                    return input_ids[start_idx:end_idx]

        # If no match is found, return an empty array
        return np.array([], dtype=np.intc)

    def __call__(
        self, input_ids: npt.NDArray[np.intc], /, **kwargs: Any
    ) -> npt.NDArray[np.intc]:
        return self.find_candidate_pred_tokens(
            input_ids=input_ids,
            max_ngram_size=self.max_ngram_size,
            num_pred_tokens=self.num_pred_tokens,
        )
