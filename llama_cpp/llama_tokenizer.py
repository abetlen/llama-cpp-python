from __future__ import annotations

import abc
from typing import (
    List,
    Optional,
    Any,
)

import llama_cpp
from llama_cpp.llama_types import List


class BaseLlamaTokenizer(abc.ABC):
    @abc.abstractmethod
    def tokenize(
        self, text: bytes, add_bos: bool = True, special: bool = True
    ) -> List[int]:
        """Tokenize the text into tokens.

        Args:
            text: The utf-8 encoded string to tokenize.
            add_bos: Whether to add a beginning of sequence token.
            special: Whether to tokenize special tokens.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def detokenize(
        self,
        tokens: List[int],
        prev_tokens: Optional[List[int]] = None,
        special: bool = False,
    ) -> bytes:
        """Detokenize the tokens into text.

        Args:
            tokens: The list of tokens to detokenize.
            prev_tokens: The list of previous tokens. Offset mapping will be performed if provided.
            special: Whether to detokenize special tokens.
        """
        raise NotImplementedError


class LlamaTokenizer(BaseLlamaTokenizer):
    def __init__(self, llama: llama_cpp.Llama):
        self._model = llama._model  # type: ignore

    def tokenize(
        self, text: bytes, add_bos: bool = True, special: bool = True
    ) -> List[int]:
        return self._model.tokenize(text, add_bos=add_bos, special=special)

    def detokenize(
        self,
        tokens: List[int],
        prev_tokens: Optional[List[int]] = None,
        special: bool = False,
    ) -> bytes:
        return self._model.detokenize(tokens, special=special)

    def encode(
        self, text: str, add_bos: bool = True, special: bool = True
    ) -> List[int]:
        return self.tokenize(
            text.encode("utf-8", errors="ignore"), add_bos=add_bos, special=special
        )

    def decode(self, tokens: List[int]) -> str:
        return self.detokenize(tokens).decode("utf-8", errors="ignore")

    @classmethod
    def from_ggml_file(cls, path: str) -> "LlamaTokenizer":
        return cls(llama_cpp.Llama(model_path=path, vocab_only=True))


class LlamaHFTokenizer(BaseLlamaTokenizer):
    def __init__(self, hf_tokenizer: Any):
        self.hf_tokenizer = hf_tokenizer

    def tokenize(
        self, text: bytes, add_bos: bool = True, special: bool = True
    ) -> List[int]:
        return self.hf_tokenizer.encode(
            text.decode("utf-8", errors="ignore"), add_special_tokens=special
        )

    def detokenize(
        self,
        tokens: List[int],
        prev_tokens: Optional[List[int]] = None,
        special: bool = False,
    ) -> bytes:
        skip_special_tokens = not special
        if prev_tokens is not None:
            text = self.hf_tokenizer.decode(
                prev_tokens + tokens, skip_special_tokens=skip_special_tokens
            ).encode("utf-8", errors="ignore")
            prev_text = self.hf_tokenizer.decode(
                prev_tokens, skip_special_tokens=skip_special_tokens
            ).encode("utf-8", errors="ignore")
            return text[len(prev_text) :]
        else:
            return self.hf_tokenizer.decode(
                tokens, skip_special_tokens=skip_special_tokens
            ).encode("utf-8", errors="ignore")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str) -> "LlamaHFTokenizer":
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "The `transformers` library is required to use the `HFTokenizer`."
                "You can install it with `pip install transformers`."
            )
        hf_tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path
        )
        return cls(hf_tokenizer)
