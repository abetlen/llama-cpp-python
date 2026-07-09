"""Test type safety validation fixes for llama_cpp.

These tests verify that parameter validation is correctly applied
in Llama.__init__, Llama.tokenize, Llama.detokenize, and
Llama.create_completion without requiring a real model file.
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# We need to mock the native library before importing llama_cpp.
# Create a mock for the native llama_cpp module.
_mock_native = MagicMock()
_mock_native.LLAMA_SPLIT_MODE_LAYER = 0
_mock_native.LLAMA_DEFAULT_SEED = -1
_mock_native.LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED = -1
_mock_native.LLAMA_POOLING_TYPE_UNSPECIFIED = -1
_mock_native.LLAMA_ATTENTION_TYPE_UNSPECIFIED = -1
_mock_native.LLAMA_MAX_DEVICES = 16
_mock_native.GGML_NUMA_STRATEGY_DISABLED = 0
_mock_native.GGML_NUMA_STRATEGY_DISTRIBUTE = 1
_mock_native.llama_model_default_params.return_value = Mock()
_mock_native.llama_log_callback = MagicMock()
_mock_native.llama_log_set = MagicMock()

# Patch the native module before llama_cpp imports it
sys.modules.setdefault('llama_cpp.llama_cpp', _mock_native)

# Now we can import
from llama_cpp.llama import Llama


class TestInitValidation:
    """Test __init__ parameter validation (llama_cpp/llama.py lines 201-209)"""

    def test_empty_model_path_raises_value_error(self):
        with pytest.raises(ValueError, match="model_path cannot be empty or None"):
            Llama(model_path="")

    def test_none_model_path_raises_value_error(self):
        with pytest.raises(ValueError, match="model_path cannot be empty or None"):
            Llama(model_path=None)

    def test_non_string_model_path_int_raises_type_error(self):
        with pytest.raises(TypeError, match="model_path must be a string"):
            Llama(model_path=123)

    def test_non_string_model_path_list_raises_type_error(self):
        with pytest.raises(TypeError, match="model_path must be a string"):
            Llama(model_path=['path'])

    def test_non_string_model_path_dict_raises_type_error(self):
        with pytest.raises(TypeError, match="model_path must be a string"):
            Llama(model_path={'path': 'value'})

    def test_n_ctx_zero_raises_value_error(self):
        with pytest.raises(ValueError, match="n_ctx must be > 0, got 0"):
            Llama(model_path="valid_path.gguf", n_ctx=0)

    def test_n_ctx_negative_raises_value_error(self):
        with pytest.raises(ValueError, match="n_ctx must be > 0, got -1"):
            Llama(model_path="valid_path.gguf", n_ctx=-1)

    def test_n_ctx_negative_large_raises_value_error(self):
        with pytest.raises(ValueError, match="n_ctx must be > 0, got -100"):
            Llama(model_path="valid_path.gguf", n_ctx=-100)

    def test_n_ctx_positive_does_not_raise_validation_error(self):
        """Verify n_ctx > 0 passes the validation check (may fail later on model load)."""
        try:
            Llama(model_path="valid_path.gguf", n_ctx=512)
        except (ValueError, TypeError) as e:
            # Check if it's NOT our validation error
            if "n_ctx must be > 0" in str(e) or "model_path must be a string" in str(e) or "model_path cannot be empty" in str(e):
                pytest.fail(f"Unexpected validation error for n_ctx=512: {e}")
            # Other ValueErrors (e.g., model path doesn't exist) are expected
        except Exception:
            # Other exceptions (e.g., model load failure) are expected
            pass

    def test_n_ctx_one_passes_validation(self):
        """Boundary: n_ctx=1 is the minimum valid value."""
        try:
            Llama(model_path="valid_path.gguf", n_ctx=1)
        except (ValueError, TypeError) as e:
            # Check if it's NOT our validation error
            if "n_ctx must be > 0" in str(e) or "model_path must be a string" in str(e) or "model_path cannot be empty" in str(e):
                pytest.fail(f"Unexpected validation error for n_ctx=1: {e}")
            # Other ValueErrors (e.g., model path doesn't exist) are expected
        except Exception:
            pass


class TestTokenizeValidation:
    """Test tokenize method parameter validation (llama_cpp/llama.py lines 631-634)"""

    def _make_model(self):
        """Create a Llama instance with mocked internals."""
        model = Llama.__new__(Llama)
        model.tokenizer_ = Mock()
        return model

    def test_int_text_raises_type_error(self):
        model = self._make_model()
        with pytest.raises(TypeError, match="text must be str or bytes, got int"):
            model.tokenize(123)

    def test_list_text_raises_type_error(self):
        model = self._make_model()
        with pytest.raises(TypeError, match="text must be str or bytes, got list"):
            model.tokenize(['text'])

    def test_none_text_raises_type_error(self):
        model = self._make_model()
        with pytest.raises(TypeError, match="text must be str or bytes, got NoneType"):
            model.tokenize(None)

    def test_dict_text_raises_type_error(self):
        model = self._make_model()
        with pytest.raises(TypeError, match="text must be str or bytes, got dict"):
            model.tokenize({'key': 'value'})

    def test_float_text_raises_type_error(self):
        model = self._make_model()
        with pytest.raises(TypeError, match="text must be str or bytes, got float"):
            model.tokenize(3.14)

    def test_str_text_is_accepted(self):
        model = self._make_model()
        model.tokenizer_.tokenize.return_value = [1, 2, 3]
        result = model.tokenize("test text")
        assert result == [1, 2, 3]
        model.tokenizer_.tokenize.assert_called_once_with("test text", True, False)

    def test_bytes_text_is_accepted(self):
        model = self._make_model()
        model.tokenizer_.tokenize.return_value = [1, 2, 3]
        result = model.tokenize(b"test text")
        assert result == [1, 2, 3]
        model.tokenizer_.tokenize.assert_called_once_with(b"test text", True, False)

    def test_empty_str_is_accepted(self):
        model = self._make_model()
        model.tokenizer_.tokenize.return_value = []
        result = model.tokenize("")
        assert result == []


class TestDetokenizeValidation:
    """Test detokenize method parameter validation (llama_cpp/llama.py lines 653-661)"""

    def _make_model(self):
        """Create a Llama instance with mocked internals."""
        model = Llama.__new__(Llama)
        model.tokenizer_ = Mock()
        return model

    def test_str_tokens_raises_type_error(self):
        model = self._make_model()
        with pytest.raises(TypeError, match="tokens must be a list, got str"):
            model.detokenize("not a list")

    def test_int_tokens_raises_type_error(self):
        model = self._make_model()
        with pytest.raises(TypeError, match="tokens must be a list, got int"):
            model.detokenize(123)

    def test_none_tokens_raises_type_error(self):
        model = self._make_model()
        with pytest.raises(TypeError, match="tokens must be a list, got NoneType"):
            model.detokenize(None)

    def test_tuple_tokens_raises_type_error(self):
        model = self._make_model()
        with pytest.raises(TypeError, match="tokens must be a list, got tuple"):
            model.detokenize((1, 2, 3))

    def test_dict_tokens_raises_type_error(self):
        model = self._make_model()
        with pytest.raises(TypeError, match="tokens must be a list, got dict"):
            model.detokenize({0: 1})

    def test_list_with_string_element_raises_type_error(self):
        model = self._make_model()
        with pytest.raises(TypeError, match="all elements in tokens must be integers"):
            model.detokenize([1, 2, "three"])

    def test_list_with_float_element_raises_type_error(self):
        model = self._make_model()
        with pytest.raises(TypeError, match="all elements in tokens must be integers"):
            model.detokenize([1, 2.5, 3])

    def test_list_with_none_element_raises_type_error(self):
        model = self._make_model()
        with pytest.raises(TypeError, match="all elements in tokens must be integers"):
            model.detokenize([1, None, 3])

    def test_valid_list_is_accepted(self):
        model = self._make_model()
        model.tokenizer_.detokenize.return_value = b"test text"
        result = model.detokenize([1, 2, 3])
        assert result == b"test text"

    def test_empty_list_is_accepted(self):
        model = self._make_model()
        model.tokenizer_.detokenize.return_value = b""
        result = model.detokenize([])
        assert result == b""


class TestCreateCompletionValidation:
    """Test create_completion method parameter validation (llama_cpp/llama.py lines 1903-1911)"""

    def _make_model(self):
        """Create a Llama instance with mocked internals."""
        model = Llama.__new__(Llama)
        # Mock _create_completion to return an iterator (generator)
        def mock_completion(*args, **kwargs):
            yield {"choices": [{"text": "test", "index": 0}]}
        model._create_completion = Mock(side_effect=mock_completion)
        return model

    def test_negative_temperature_raises_value_error(self):
        model = self._make_model()
        with pytest.raises(ValueError, match=r"temperature must be >= 0\.0, got -0\.5"):
            model.create_completion("test", temperature=-0.5)

    def test_negative_temperature_minus_one_raises_value_error(self):
        model = self._make_model()
        with pytest.raises(ValueError, match=r"temperature must be >= 0\.0, got -1\.0"):
            model.create_completion("test", temperature=-1.0)

    def test_top_p_above_one_raises_value_error(self):
        model = self._make_model()
        with pytest.raises(ValueError, match=r"top_p must be in \[0\.0, 1\.0\], got 1\.5"):
            model.create_completion("test", top_p=1.5)

    def test_top_p_above_two_raises_value_error(self):
        model = self._make_model()
        with pytest.raises(ValueError, match=r"top_p must be in \[0\.0, 1\.0\], got 2\.0"):
            model.create_completion("test", top_p=2.0)

    def test_top_p_negative_raises_value_error(self):
        model = self._make_model()
        with pytest.raises(ValueError, match=r"top_p must be in \[0\.0, 1\.0\], got -0\.5"):
            model.create_completion("test", top_p=-0.5)

    def test_negative_top_k_raises_value_error(self):
        model = self._make_model()
        with pytest.raises(ValueError, match="top_k must be >= 0, got -1"):
            model.create_completion("test", top_k=-1)

    def test_negative_top_k_large_raises_value_error(self):
        model = self._make_model()
        with pytest.raises(ValueError, match="top_k must be >= 0, got -10"):
            model.create_completion("test", top_k=-10)

    def test_valid_params_do_not_raise(self):
        model = self._make_model()
        model.create_completion("test", temperature=0.8, top_p=0.95, top_k=40)
        model._create_completion.assert_called_once()

    def test_boundary_temperature_zero(self):
        model = self._make_model()
        model.create_completion("test", temperature=0.0)
        model._create_completion.assert_called_once()

    def test_boundary_top_p_zero(self):
        model = self._make_model()
        model.create_completion("test", top_p=0.0)
        model._create_completion.assert_called_once()

    def test_boundary_top_p_one(self):
        model = self._make_model()
        model.create_completion("test", top_p=1.0)
        model._create_completion.assert_called_once()

    def test_boundary_top_k_zero(self):
        model = self._make_model()
        model.create_completion("test", top_k=0)
        model._create_completion.assert_called_once()

    def test_multiple_invalid_params_first_fails(self):
        """When multiple params are invalid, the first check (temperature) fails."""
        model = self._make_model()
        with pytest.raises(ValueError, match="temperature"):
            model.create_completion("test", temperature=-1.0, top_p=2.0, top_k=-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
