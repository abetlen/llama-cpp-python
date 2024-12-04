"""Internal module use at your own risk

This module provides a minimal interface for working with ggml tensors from llama-cpp-python
"""
import os
import pathlib

import llama_cpp._ctypes_extensions as ctypes_ext

libggml_base_path = pathlib.Path(os.path.abspath(os.path.dirname(__file__))) / "lib"
libggml = ctypes_ext.load_shared_library("ggml", libggml_base_path)

