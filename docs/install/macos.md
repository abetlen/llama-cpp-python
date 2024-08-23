---
title: MacOS Install with Metal GPU
---

**(1) Make sure you have xcode installed... at least the command line parts**
```
# check the path of your xcode install 
xcode-select -p

# xcode installed returns
# /Applications/Xcode-beta.app/Contents/Developer

# if xcode is missing then install it... it takes ages;
xcode-select --install
```

**(2) Install the conda version for MacOS that supports Metal GPU**
```
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
bash Miniforge3-MacOSX-arm64.sh
```

**(3) Make a conda environment**
```
conda create -n llama python=3.9.16
conda activate llama
```

**(4) Install the LATEST llama-cpp-python...which happily supports MacOS Metal GPU as of version 0.1.62**  
    *(you needed xcode installed in order pip to build/compile the C++ code)*
```
pip uninstall llama-cpp-python -y
CMAKE_ARGS="-DGGML_METAL=on" pip install -U llama-cpp-python --no-cache-dir
pip install 'llama-cpp-python[server]'

# you should now have llama-cpp-python v0.1.62 or higher installed
llama-cpp-python         0.1.68

```

**(5) Download a v3 gguf v2 model**
 - **ggufv2**
 - file name ends with **Q4_0.gguf** - indicating it is 4bit quantized, with quantisation method 0

https://huggingface.co/TheBloke/CodeLlama-7B-GGUF


**(6) run the llama-cpp-python API server with MacOS Metal GPU support**
```
# config your ggml model path
# make sure it is gguf v2
# make sure it is q4_0
export MODEL=[path to your llama.cpp ggml models]]/[ggml-model-name]]Q4_0.gguf
python3 -m llama_cpp.server --model $MODEL  --n_gpu_layers 1
```

***Note:** If you omit the `--n_gpu_layers 1` then CPU will be used*


