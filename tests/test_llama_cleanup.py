import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("clear_globals", ["none", "bindings", "modules"])
@pytest.mark.parametrize("cleanup", ["close", "gc"])
def test_cleanup_after_module_teardown(clear_globals, cleanup):
    # Isolate cleared module globals and native resource lifetimes from pytest.
    script = textwrap.dedent(
        """
        import ctypes
        import gc
        import sys
        from collections import Counter

        import llama_cpp.llama as llama_module
        import llama_cpp.llama_cpp as bindings
        import llama_cpp._internals as internals

        calls = []
        errors = []
        sys.unraisablehook = errors.append

        def record(name, function):
            def wrapped(*args):
                calls.append((name, args[1] if name == 'remove' else None))
                return function(*args)
            return wrapped

        free_functions = {
            'llama_model_free': 'model',
            'llama_free': 'context',
            'llama_batch_free': 'batch',
            'llama_sampler_free': 'sampler',
            'llama_sampler_chain_remove': 'remove',
            'llama_adapter_lora_free': 'adapter',
        }
        # Exercise the adapter cleanup path without downloading a LoRA model.
        bindings.llama_adapter_lora_init = lambda *args: ctypes.pointer(ctypes.c_void_p(1))
        bindings.llama_set_adapters_lora = lambda *args: 0
        bindings.llama_adapter_lora_free = lambda *args: None
        for function, name in free_functions.items():
            setattr(bindings, function, record(name, getattr(bindings, function)))

        model = llama_module.Llama(
            model_path='./vendor/llama.cpp/models/ggml-vocab-llama-spm.gguf',
            vocab_only=True,
            lora_path='test-adapter',
            verbose=False,
        )
        sampler = internals.LlamaSampler()
        sampler.add_greedy()
        clone = sampler.clone()
        sampler.add_custom(lambda candidates: None)
        sampler.add_custom(lambda candidates: None)

        clear_globals, cleanup = sys.argv[1:]
        if clear_globals == 'bindings':
            for function in free_functions:
                setattr(bindings, function, None)
        elif clear_globals == 'modules':
            internals.llama_cpp = None
            llama_module.llama_cpp = None

        if cleanup == 'close':
            clone.close()
            sampler.close()
            model.close()
            # Explicit close and subsequent finalization must release once.
            clone.close()
            sampler.close()
            model.close()
            assert clone.sampler is None
            assert sampler.sampler is None
            assert model._model.model is None
            assert model._ctx.ctx is None
            assert model._batch.batch is None
            assert model._lora_adapter is None

        del clone, sampler, model
        gc.collect()

        assert not errors, [str(error.exc_value) for error in errors]
        assert Counter(name for name, _ in calls) == {
            'model': 1, 'context': 1, 'batch': 1,
            'sampler': 2, 'remove': 2, 'adapter': 1,
        }, calls
        assert [index for name, index in calls if name == 'remove'] == [2, 1]
        assert calls.index(('context', None)) < calls.index(('model', None))
        assert calls.index(('adapter', None)) < calls.index(('model', None))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, clear_globals, cleanup],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Exception ignored" not in result.stderr
