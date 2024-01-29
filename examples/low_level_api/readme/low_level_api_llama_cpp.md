# Low-Level API for Llama_cpp

## Overview
This Python script, low_level_api_llama_cpp.py, demonstrates the implementation of a low-level API for interacting with the llama_cpp library. The script defines an inference that generates embeddings based on a given prompt using .gguf model.

### Prerequisites
Before running the script, ensure that you have the following dependencies installed:

.    Python 3.6 or higher
.    llama_cpp: A C++ library for working with .gguf model
.    NumPy: A fundamental package for scientific computing with Python
.    multiprocessing: A Python module for parallel computing

### Usage
install depedencies:
```bash
python -m pip install llama-cpp-python ctypes os multiprocessing
```
Run the script:
```bash
python low_level_api_llama_cpp.py
```

## Code Structure
The script is organized as follows:

### . Initialization:
        Load the model from the specified path.
        Create a context for model evaluation.

### . Tokenization:
        Tokenize the input prompt using the llama_tokenize function.
        Prepare the input tokens for model evaluation.

### . Inference:
        Perform model evaluation to generate responses.
        Sample from the model's output using various strategies (top-k, top-p, temperature).

### . Output:
        Print the generated tokens and the corresponding decoded text.

### .Cleanup:
        Free resources and print timing information.

## Configuration
Customize the inference behavior by adjusting the following variables:

#### . N_THREADS: Number of CPU threads to use for model evaluation.
#### . MODEL_PATH: Path to the model file.
#### . prompt: Input prompt for the chatbot.

## Notes
.    Ensure that the llama_cpp library is built and available in the system. Follow the instructions in the llama_cpp repository for building and installing the library.

.    This script is designed to work with the .gguf model and may require modifications for compatibility with other models.

## Acknowledgments
This code is based on the llama_cpp library developed by the community. Special thanks to the contributors for their efforts.

## License
This project is licensed under the MIT License - see the LICENSE file for details.