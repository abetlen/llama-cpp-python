# Install Docker Server

**Note #1:** This was tested with Docker running on Linux. If you can get it working on Windows or MacOS, please update this `README.md` with a PR!

[Install Docker Engine](https://docs.docker.com/engine/install)

**Note #2:** NVidia GPU CuBLAS support requires a NVidia GPU with sufficient VRAM (approximately as much as the size in the table below) and Docker NVidia support (see [container-toolkit/install-guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))

# Simple Dockerfiles for building the llama-cpp-python server with external model bin files
## openblas_simple - a simple Dockerfile for non-GPU OpenBLAS, where the model is located outside the Docker image
```
cd ./openblas_simple
docker build -t openblas_simple .
docker run -e USE_MLOCK=0 -e MODEL=/var/model/<model-path> -v <model-root-path>:/var/model -t openblas_simple
```
where `<model-root-path>/<model-path>` is the full path to the model file on the Docker host system.

## cuda_simple - a simple Dockerfile for CUDA accelerated CuBLAS, where the model is located outside the Docker image
```
cd ./cuda_simple
docker build -t cuda_simple .
docker run -e USE_MLOCK=0 -e MODEL=/var/model/<model-path> -v <model-root-path>:/var/model -t cuda_simple
```
where `<model-root-path>/<model-path>` is the full path to the model file on the Docker host system.

# "Open-Llama-in-a-box"
## Download an Apache V2.0 licensed 3B paramter Open Llama model and install into a Docker image that runs an OpenBLAS-enabled llama-cpp-python server
```
$ cd ./open_llama
./build.sh
./start.sh
```

# Manually choose your own Llama model from Hugging Face
`python3 ./hug_model.py -a TheBloke -t llama`
You should now have a model in the current directory and `model.bin` symlinked to it for the subsequent Docker build and copy step. e.g.
```
docker $ ls -lh *.bin
-rw-rw-r-- 1 user user 4.8G May 23 18:30 <downloaded-model-file>q5_1.bin
lrwxrwxrwx 1 user user   24 May 23 18:30 model.bin -> <downloaded-model-file>q5_1.bin
```
**Note #1:** Make sure you have enough disk space to download the model. As the model is then copied into the image you will need at least
**TWICE** as much disk space as the size of the model:

| Model |  Quantized size |
|------:|----------------:|
|    3B |            3 GB |
|    7B |            5 GB |
|   13B |           10 GB |
|   33B |           25 GB |
|   65B |           50 GB |

**Note #2:** If you want to pass or tune additional parameters, customise `./start_server.sh` before running `docker build ...`

## Use OpenBLAS
Use if you don't have a NVidia GPU. Defaults to `python:3-slim-bullseye` Docker base image and OpenBLAS:
### Build:
`docker build -t openblas .`
### Run:
`docker run --cap-add SYS_RESOURCE -t openblas`

## Use CuBLAS
### Build:
`docker build --build-arg IMAGE=nvidia/cuda:12.1.1-devel-ubuntu22.04 -t cublas .`
### Run:
`docker run --cap-add SYS_RESOURCE -t cublas`
