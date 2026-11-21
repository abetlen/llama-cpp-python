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

    def __init__(self, max_ngram_size: int = 2, num_pred_tokens: int = 10):
        self.max_ngram_size = max_ngram_size
        self.num_pred_tokens = num_pred_tokens

    @staticmethod
    def find_candidate_pred_tokens(
        input_ids: npt.NDArray[np.intc],
        max_ngram_size: int,
        num_pred_tokens: int,
    ):
        input_length = input_ids.shape[0]
        use_slices = (
            type(input_ids) is np.ndarray
            and input_ids.ndim == 1
            and input_ids.dtype.kind in "iu"
            and type(num_pred_tokens) is int
        )

        native_indices = use_slices and -input_length <= num_pred_tokens <= input_length

        for ngram_size in range(min(max_ngram_size, input_length - 1), 0, -1):
            if use_slices:
                window_count = input_length - ngram_size + 1
                matches = input_ids[:window_count] == input_ids[-ngram_size]
                if native_indices and ngram_size > 1 and not matches[:-1].any():
                    continue
                for offset in range(1, ngram_size):
                    matches &= (
                        input_ids[offset : offset + window_count]
                        == input_ids[offset - ngram_size]
                    )
                # The suffix itself always matches for integer arrays.
                first_match = matches.argmax()
                match_indices = (int(first_match) if native_indices else first_match,)
            else:
                windows = np.lib.stride_tricks.sliding_window_view(
                    input_ids, (ngram_size,)
                )
                matches = np.all(windows == input_ids[-ngram_size:], axis=1)
                match_indices = np.nonzero(matches)[0]

            for idx in match_indices:
                start_idx = idx + ngram_size
                end_idx = min(start_idx + num_pred_tokens, input_length)
                if start_idx < end_idx:
                    return input_ids[start_idx:end_idx]

        return np.array([], dtype=np.intc)

    def __call__(
        self, input_ids: npt.NDArray[np.intc], /, **kwargs: Any
    ) -> npt.NDArray[np.intc]:
        return self.find_candidate_pred_tokens(
            input_ids=input_ids,
            max_ngram_size=self.max_ngram_size,
            num_pred_tokens=self.num_pred_tokens,
        )
