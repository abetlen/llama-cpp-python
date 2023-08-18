### Install Docker Server
> [!IMPORTANT]  
> This was tested with Docker running on Linux. <br>If you can get it working on Windows or MacOS, please update this `README.md` with a PR!<br>

[Install Docker Engine](https://docs.docker.com/engine/install)


## Simple Dockerfiles for building the llama-cpp-python server with external model bin files
### openblas_simple
A simple Dockerfile for non-GPU OpenBLAS, where the model is located outside the Docker image:
```
cd ./openblas_simple
docker build -t openblas_simple .
docker run --cap-add SYS_RESOURCE -e USE_MLOCK=0 -e MODEL=/var/model/<model-path> -v <model-root-path>:/var/model -t openblas_simple
```
where `<model-root-path>/<model-path>` is the full path to the model file on the Docker host system.

### cuda_simple
> [!WARNING]  
> Nvidia GPU CuBLAS support requires an Nvidia GPU with sufficient VRAM (approximately as much as the size in the table below) and Docker Nvidia support (see [container-toolkit/install-guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)) <br>

A simple Dockerfile for CUDA-accelerated CuBLAS, where the model is located outside the Docker image:

```
cd ./cuda_simple
docker build -t cuda_simple .
docker run --gpus=all --cap-add SYS_RESOURCE -e USE_MLOCK=0 -e MODEL=/var/model/<model-path> -v <model-root-path>:/var/model -t cuda_simple
```
where `<model-root-path>/<model-path>` is the full path to the model file on the Docker host system.

--------------------------------------------------------------------------

### "Open-Llama-in-a-box"
Download an Apache V2.0 licensed 3B params Open LLaMA model and install into a Docker image that runs an OpenBLAS-enabled llama-cpp-python server:
```
$ cd ./open_llama
./build.sh
./start.sh
```

### Manually choose your own Llama model from Hugging Face
`python3 ./hug_model.py -a TheBloke -t llama`
You should now have a model in the current directory and `model.bin` symlinked to it for the subsequent Docker build and copy step. e.g.
```
docker $ ls -lh *.bin
-rw-rw-r-- 1 user user 4.8G May 23 18:30 <downloaded-model-file>q5_1.bin
lrwxrwxrwx 1 user user   24 May 23 18:30 model.bin -> <downloaded-model-file>q5_1.bin
```

> [!NOTE]  
> Make sure you have enough disk space to download the model. As the model is then copied into the image you will need at least
**TWICE** as much disk space as the size of the model:<br>

| Model |  Quantized size |
|------:|----------------:|
|    3B |            3 GB |
|    7B |            5 GB |
|   13B |           10 GB |
|   33B |           25 GB |
|   65B |           50 GB |


> [!NOTE]  
> If you want to pass or tune additional parameters, customise `./start_server.sh` before running `docker build ...`
