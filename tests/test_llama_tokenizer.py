from llama_cpp.llama_tokenizer import LlamaHFTokenizer


class SyntheticHFTokenizer:
    bos_token_id = 1
    eos_token_id = 2

    def encode(self, text, add_special_tokens=True):
        tokens = [self.bos_token_id, 3] if text.startswith("<bos>") else [3]
        if add_special_tokens:
            return [self.bos_token_id] + tokens + [self.eos_token_id]
        return tokens


def test_hf_tokenizer_respects_add_bos_false():
    tokenizer = LlamaHFTokenizer(SyntheticHFTokenizer())

    assert tokenizer.tokenize(b"prompt", add_bos=False, special=True) == [3, 2]


def test_hf_tokenizer_adds_bos_when_special_tokens_are_disabled():
    tokenizer = LlamaHFTokenizer(SyntheticHFTokenizer())

    assert tokenizer.tokenize(b"prompt", add_bos=True, special=False) == [1, 3]


def test_hf_tokenizer_preserves_explicit_bos_when_add_bos_is_false():
    tokenizer = LlamaHFTokenizer(SyntheticHFTokenizer())

    assert tokenizer.tokenize(b"<bos>prompt", add_bos=False, special=True) == [1, 3, 2]


def test_hf_tokenizer_does_not_duplicate_bos_when_special_tokens_add_it():
    tokenizer = LlamaHFTokenizer(SyntheticHFTokenizer())

    assert tokenizer.tokenize(b"prompt", add_bos=True, special=True) == [1, 3, 2]
