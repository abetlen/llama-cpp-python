# Dockerfiles for building the llama-cpp-python server
- `Dockerfile.openblas_simple` - a simple Dockerfile for non-GPU OpenBLAS
- `Dockerfile.cuda_simple` - a simple Dockerfile for CUDA accelerated CuBLAS
- `hug_model.py` - a Python utility for interactively choosing and downloading the latest `5_1` quantized models from [huggingface.co/TheBloke]( https://huggingface.co/TheBloke)
- `Dockerfile` - a single OpenBLAS and CuBLAS combined Dockerfile that automatically installs a previously downloaded model `model.bin`
 
# Get model from Hugging Face
`python3 ./hug_model.py`

You should now have a model in the current directory and `model.bin` symlinked to it for the subsequent Docker build and copy step. e.g.
```
docker $ ls -lh *.bin
-rw-rw-r-- 1 user user 4.8G May 23 18:30 <downloaded-model-file>.q5_1.bin
lrwxrwxrwx 1 user user   24 May 23 18:30 model.bin -> <downloaded-model-file>.q5_1.bin
```
**Note #1:** Make sure you have enough disk space to download the model. As the model is then copied into the image you will need at least
**TWICE** as much disk space as the size of the model:

| Model |  Quantized size |
|------:|----------------:|
|    7B |            5 GB |
|   13B |           10 GB |
|   30B |           25 GB |
|   65B |           50 GB |

**Note #2:** If you want to pass or tune additional parameters, customise `./start_server.sh` before running `docker build ...`

# Install Docker Server

**Note #3:** This was tested with Docker running on Linux. If you can get it working on Windows or MacOS, please update this `README.md` with a PR!

[Install Docker Engine](https://docs.docker.com/engine/install)

# Use OpenBLAS
Use if you don't have a NVidia GPU. Defaults to `python:3-slim-bullseye` Docker base image and OpenBLAS:
## Build:
`docker build --build-arg -t openblas .`
## Run:
`docker run --cap-add SYS_RESOURCE -t openblas`

# Use CuBLAS
Requires a NVidia GPU with sufficient VRAM (approximately as much as the size above) and Docker NVidia support (see [container-toolkit/install-guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))
## Build:
`docker build --build-arg IMAGE=nvidia/cuda:12.1.1-devel-ubuntu22.04 -t cublas .`
## Run:
`docker run --cap-add SYS_RESOURCE -t cublas`
