ARG CUDA_VERSION="12.1.1"
ARG OS="ubuntu22.04"

ARG CUDA_BUILDER_IMAGE="${CUDA_VERSION}-devel-${OS}"
ARG CUDA_RUNTIME_IMAGE="${CUDA_VERSION}-runtime-${OS}"
FROM nvidia/cuda:${CUDA_BUILDER_IMAGE} as builder

RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y git build-essential \
    python3 python3-pip python3-venv gcc wget \
    ocl-icd-opencl-dev opencl-headers clinfo \
    libclblast-dev libopenblas-dev \
    && mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

WORKDIR /llama_cpp_python

COPY . .

# setting build related env vars
ENV CUDA_DOCKER_ARCH=all
ENV LLAMA_CUBLAS=1

# Install depencencies
RUN python3 -m pip install --upgrade pip
RUN python3 -m venv venv
# RUN python3 -m pip install --upgrade pip pytest cmake scikit-build setuptools fastapi uvicorn sse-starlette pydantic-settings starlette-context

# Install llama-cpp-python (build with cuda)
RUN CMAKE_ARGS="-DLLAMA_CUBLAS=on" venv/bin/pip install .[server]
# RUN make clean

FROM nvidia/cuda:${CUDA_RUNTIME_IMAGE} as runtime

# We need to set the host to 0.0.0.0 to allow outside access
ENV HOST 0.0.0.0
ENV CUDA_DOCKER_ARCH=all

RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y python3 python3-pip python3-venv

WORKDIR /llama_cpp_python

COPY --from=builder /llama_cpp_python/venv venv

# Run the server
CMD venv/bin/python3 -m llama_cpp.server