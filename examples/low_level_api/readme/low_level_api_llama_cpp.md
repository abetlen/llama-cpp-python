# Generate text with the low-level llama.cpp API

Install the Python bindings and run the example from the repository root with a
GGUF text-generation model:

```bash
python -m pip install llama-cpp-python
MODEL=/path/to/model.gguf python examples/low_level_api/low_level_api_llama_cpp.py
```

The script prints the prompt, streams up to 20 new tokens, and reports context
timings. Edit `PROMPT` and `N_PREDICT` in the script to try a different
completion. The prompt and completion share the context window.

The example tokenizes the prompt with the model vocabulary, then passes batches
for a single sequence to `llama_decode`. The final token of each batch requests
logits. A sampler chain applies top-k, top-p, and temperature sampling, then
draws the next token from the resulting distribution. Generation stops at an
end-of-generation token or the token limit. Token pieces are decoded
incrementally so a UTF-8 character can span multiple tokens.
