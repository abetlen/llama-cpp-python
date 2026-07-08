"""Test for issue #1762 fix - auto-adjust n_ubatch for embedding"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from llama_cpp import Llama


def test_embed_auto_adjusts_n_ubatch():
    """Test that embed() automatically adjusts n_ubatch when n_batch > 512"""
    # Mock the Llama class
    with patch('llama_cpp.Llama.__init__', return_value=None):
        llama = Llama.__new__(Llama)
        
        # Setup mock attributes
        llama.n_batch = 4096
        llama.context_params = Mock()
        llama.context_params.n_ubatch = 512  # Default value
        llama.context_params.embeddings = True
        llama.context_params.n_seq_max = 1
        llama._model = Mock()
        llama._model.n_embd = Mock(return_value=768)
        llama._ctx = Mock()
        llama.verbose = False
        
        # Mock set_n_ubatch method
        llama.set_n_ubatch = Mock()
        
        # Call embed - should auto-adjust n_ubatch
        try:
            llama.embed("test input")
        except Exception:
            pass  # We expect it to fail later, we just want to check n_ubatch adjustment
        
        # Verify set_n_ubatch was called with n_batch value
        llama.set_n_ubatch.assert_called_once_with(4096)


def test_embed_no_adjust_when_n_ubatch_sufficient():
    """Test that embed() doesn't adjust n_ubatch when it's already >= n_batch"""
    with patch('llama_cpp.Llama.__init__', return_value=None):
        llama = Llama.__new__(Llama)
        
        # Setup mock attributes with sufficient n_ubatch
        llama.n_batch = 4096
        llama.context_params = Mock()
        llama.context_params.n_ubatch = 4096  # Already sufficient
        llama.context_params.embeddings = True
        llama.context_params.n_seq_max = 1
        llama._model = Mock()
        llama._model.n_embd = Mock(return_value=768)
        llama._ctx = Mock()
        llama.verbose = False
        
        # Mock set_n_ubatch method
        llama.set_n_ubatch = Mock()
        
        # Call embed - should NOT adjust n_ubatch
        try:
            llama.embed("test input")
        except Exception:
            pass
        
        # Verify set_n_ubatch was NOT called
        llama.set_n_ubatch.assert_not_called()


if __name__ == "__main__":
    test_embed_auto_adjusts_n_ubatch()
    test_embed_no_adjust_when_n_ubatch_sufficient()
    print("All tests passed!")
