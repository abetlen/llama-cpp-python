"""Regression tests for logits processors during speculative decoding.

``Llama.generate`` evaluates an accepted token together with the draft block,
so ``input_ids`` already holds unverified draft tokens when the next position
is sampled. Processors must see only the prompt plus tokens accepted so far.
That is the same prefix greedy decoding passes when no draft model is used.

The tests below drive ``generate`` with a stub sampler and do not need a GGUF.
Manual check against the report in abetlen/llama-cpp-python#2393 (CPU, a few
seconds on ``stories260K.gguf``):

```bash
pip install huggingface_hub
python - <<'PY'
import numpy as np
from huggingface_hub import hf_hub_download
from llama_cpp import Llama, LogitsProcessorList
from llama_cpp.llama_speculative import LlamaPromptLookupDecoding

path = hf_hub_download("ggml-org/models", "tinyllamas/stories260K.gguf")
PROMPT = b"Once upon a time, there was a little girl. " * 4 + b"Once upon a time,"

def run(draft, use_processor=True, n_new=48):
    llm = Llama(
        model_path=path,
        n_ctx=512,
        logits_all=True,
        verbose=False,
        draft_model=LlamaPromptLookupDecoding(num_pred_tokens=4) if draft else None,
    )
    prompt, out, calls, wrong_prefix = llm.tokenize(PROMPT), [], 0, 0

    def parity(input_ids, scores):
        nonlocal calls, wrong_prefix
        calls += 1
        wrong_prefix += list(input_ids) != prompt + out
        if len(input_ids) % 2 == 0:
            scores[0::2] = -np.inf
        return scores

    processor = LogitsProcessorList([parity]) if use_processor else None
    for tok in llm.generate(prompt, temp=0.0, logits_processor=processor):
        out.append(tok)
        if len(out) == n_new:
            break
    violations = sum(
        t % 2 == 0 and (len(prompt) + i) % 2 == 0 for i, t in enumerate(out)
    )
    return out, calls, wrong_prefix, violations

assert run(False, use_processor=False)[0] == run(True, use_processor=False)[0]
for draft in (False, True):
    out, calls, wrong_prefix, violations = run(draft)
    assert calls == 48
    assert wrong_prefix == 0
    assert violations == 0
assert run(False)[0] == run(True)[0]
print("prompt-lookup logits processor prefixes match greedy decoding")
PY
```
"""

import contextlib
import ctypes

import numpy as np

import llama_cpp
import llama_cpp.llama as llama_module
from llama_cpp.llama_speculative import LlamaPromptLookupDecoding


def _bare_llama() -> llama_cpp.Llama:
    llm = llama_cpp.Llama.__new__(llama_cpp.Llama)
    # ``__del__`` closes this stack. A real model sets it in ``__init__``.
    llm._stack = contextlib.ExitStack()
    return llm


def test_logits_processor_prefix_len_hides_only_unverified_suffix():
    llm = _bare_llama()
    llm.n_tokens = 10

    assert llm._logits_processor_prefix_len(None) == 10
    assert llm._logits_processor_prefix_len(-1) == 10
    # Last evaluated position: same prefix greedy decoding already passes.
    assert llm._logits_processor_prefix_len(9) == 10
    assert llm._logits_processor_prefix_len(4) == 5
    # Past the buffer: do not read uninitialized tokens.
    assert llm._logits_processor_prefix_len(100) == 10


class _Context:
    def __init__(self):
        self.ctx = object()

    def kv_cache_seq_rm(self, *args, **kwargs):
        return True


class _ScriptedSampler:
    """Stand-in for ``LlamaSampler`` that runs the real processor callback."""

    def __init__(self):
        self.apply_func = None
        self._candidates = None

    def add_custom(self, apply_func):
        self.apply_func = apply_func

    def add_penalties(self, **kwargs):
        return None

    def add_greedy(self):
        return None

    def sample(self, ctx, idx):
        del ctx, idx
        size = 8
        data = (llama_cpp.llama_token_data * size)()
        for token_id in range(size):
            data[token_id].id = token_id
            # Prefer the highest id until a processor masks logits.
            data[token_id].logit = float(token_id)
            data[token_id].p = 0.0
        array = llama_cpp.llama_token_data_array()
        array.data = data
        array.size = size
        array.selected = -1
        array.sorted = False
        # Keep the buffer alive across the callback.
        self._candidates = data
        if self.apply_func is not None:
            self.apply_func(ctypes.pointer(array))
        logits = np.array([data[i].logit for i in range(size)], dtype=np.float32)
        return int(np.argmax(logits))


def _install_sampler(monkeypatch):
    monkeypatch.setattr(llama_module.internals, "LlamaSampler", _ScriptedSampler)


def _llama(draft):
    llm = _bare_llama()
    n_ctx = 64
    llm.n_tokens = 0
    llm._requires_eval = True
    llm._n_ctx = n_ctx
    llm._n_vocab = 8
    llm.last_n_tokens_size = 64
    llm._is_recurrent = False
    llm._is_hybrid = False
    llm._sampler = None
    llm._processor_sample_idx = None
    llm.input_ids = np.zeros((n_ctx,), dtype=np.intc)
    llm.scores = np.zeros((n_ctx, llm._n_vocab), dtype=np.single)
    llm._ctx = _Context()
    llm.draft_model = (
        LlamaPromptLookupDecoding(max_ngram_size=2, num_pred_tokens=2)
        if draft
        else None
    )

    def eval_tokens(tokens):
        batch = list(tokens)
        n_past = llm.n_tokens
        llm.input_ids[n_past : n_past + len(batch)] = batch
        llm.n_tokens += len(batch)
        llm._requires_eval = False

    llm.eval = eval_tokens
    return llm


def _run(llm, prompt, n_new):
    seen = []
    buffered = []

    def choose_from_prefix_length(input_ids, scores):
        # Token id is a pure function of the prefix length, in 1..5.
        seen.append([int(token) for token in input_ids])
        buffered.append(int(llm.n_tokens))
        scores[:] = -np.inf
        scores[(len(input_ids) % 5) + 1] = 10.0
        return scores

    processor = llama_cpp.LogitsProcessorList([choose_from_prefix_length])
    out = []
    for token in llm.generate(prompt, temp=0.0, logits_processor=processor):
        out.append(int(token))
        if len(out) == n_new:
            break
    return out, seen, buffered


def test_generate_logits_processor_sees_accepted_prefix_only(monkeypatch):
    _install_sampler(monkeypatch)
    # The first accepted token repeats an ngram, so prompt lookup drafts ahead.
    prompt = [4, 1, 2, 3, 4]
    n_new = 8

    plain = _llama(draft=False)
    plain_out, plain_seen, plain_buffered = _run(plain, prompt, n_new)
    drafted = _llama(draft=True)
    draft_out, draft_seen, draft_buffered = _run(drafted, prompt, n_new)

    assert plain_out == draft_out
    assert plain_seen == draft_seen
    assert len(plain_out) == n_new

    for generated, prefix in enumerate(plain_seen):
        assert prefix == prompt + plain_out[:generated]
        assert len(prefix) == plain_buffered[generated]

    for generated, prefix in enumerate(draft_seen):
        assert prefix == prompt + draft_out[:generated]
        # The evaluated buffer still contains the draft suffix here.
        assert len(prefix) <= draft_buffered[generated]

    assert any(
        len(prefix) < n_tokens for prefix, n_tokens in zip(draft_seen, draft_buffered)
    )
    # The length-derived token is what the callback wrote, not the raw argmax.
    assert plain_out[0] == (len(prompt) % 5) + 1
    assert plain_out[0] != 7


def test_min_tokens_processor_ignores_unverified_drafts(monkeypatch):
    """``MinTokensLogitsProcessor`` must not treat draft tokens as generated."""
    _install_sampler(monkeypatch)
    prompt = [4, 1, 2, 3, 4]
    eos = 7
    llm = _llama(draft=True)
    accepted = []
    masked = []
    min_tokens = llama_cpp.MinTokensLogitsProcessor(min_tokens=3, token_eos=eos)

    def observe(input_ids, scores):
        accepted.append(len(input_ids) - len(prompt))
        masked.append(bool(np.isneginf(scores[eos])))
        scores[:] = -np.inf
        scores[(len(input_ids) % 5) + 1] = 10.0
        return scores

    processor = llama_cpp.LogitsProcessorList([min_tokens, observe])
    out = []
    for token in llm.generate(prompt, temp=0.0, logits_processor=processor):
        out.append(int(token))
        if len(out) == 4:
            break

    # One call per accepted token. Drafts must not advance the count to 3 early.
    assert accepted == [0, 1, 2, 3]
    assert masked == [True, True, True, False]
    assert out[0] == (len(prompt) % 5) + 1


def test_generate_without_processor_is_unchanged_by_draft_tokens(monkeypatch):
    """No callback is installed, so a draft model must not change scripted output."""

    script = [1, 0, 1, 0, 2, 2]

    class Scripted(_ScriptedSampler):
        def __init__(self):
            super().__init__()
            self._script = list(script)

        def sample(self, ctx, idx):
            del ctx, idx
            if self.apply_func is not None:
                raise AssertionError(
                    "logits processor callback should not be installed"
                )
            return self._script.pop(0)

    monkeypatch.setattr(llama_module.internals, "LlamaSampler", Scripted)
    prompt = [1, 0, 1, 0, 1, 0]
    n_new = len(script)

    def collect(draft):
        llm = _llama(draft)
        out = []
        for token in llm.generate(prompt, temp=0.0, logits_processor=None):
            out.append(int(token))
            if len(out) == n_new:
                break
        return out

    assert collect(False) == script
    assert collect(True) == script
