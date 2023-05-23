# Get model from Hugging Face
`python3 ./hug_model.py`

You should now have a model in the current directory and model.bin symlinked to it for the subsequent Docker build and copy step. e.g.
```
docker $ ls -lh *.bin
-rw-rw-r-- 1 user user 4.8G May 23 18:30 llama-7b.ggmlv3.q5_1.bin
lrwxrwxrwx 1 user user   24 May 23 18:30 model.bin -> <downloaded-model-file>.q5_1.bin
```
- Note #1: Make sure you have enough disk space to d/l the model. As the model is then copied into the image you will need at least
**TWICE** as much disk space as the size of the model:

| Model |  Quantized size |
|------:|----------------:|
|    7B |            5 GB |
|   13B |           10 GB |
|   30B |           25 GB |
|   65B |           50 GB |

- Note #2: If you want to pass or tune additional parameters, customise `./start_server.sh` before running `docker build ...`

# Use OpenBLAS (No NVidia GPU, defaults to `python:3-slim-bullseye` Docker base image)
## Build:
`docker build --build-arg -t openblas .`
## Run:
`docker run --cap-add SYS_RESOURCE -t openblas`

# Use CuBLAS
Requires NVidia GPU and Docker NVidia support (see https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
## Build:
`docker build --build-arg IMAGE=nvidia/cuda:12.1.1-devel-ubuntu22.04 -t opencuda .`
## Run:
`docker run --cap-add SYS_RESOURCE -t cublas`
