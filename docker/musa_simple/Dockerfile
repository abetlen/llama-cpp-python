ARG MUSA_IMAGE="rc3.1.0-devel-ubuntu22.04"
FROM mthreads/musa:${MUSA_IMAGE}

# We need to set the host to 0.0.0.0 to allow outside access
ENV HOST 0.0.0.0

RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y git build-essential \
    python3 python3-pip gcc wget \
    ocl-icd-opencl-dev opencl-headers clinfo \
    libclblast-dev libopenblas-dev \
    && mkdir -p /etc/OpenCL/vendors && cp /driver/etc/OpenCL/vendors/MT.icd /etc/OpenCL/vendors/MT.icd

COPY . .

# setting build related env vars
ENV MUSA_DOCKER_ARCH=all
ENV GGML_MUSA=1

# Install dependencies
RUN python3 -m pip install --upgrade pip pytest cmake scikit-build setuptools fastapi uvicorn sse-starlette pydantic-settings starlette-context

# Install llama-cpp-python (build with musa)
RUN CMAKE_ARGS="-DGGML_MUSA=on" pip install llama-cpp-python

# Run the server
CMD python3 -m llama_cpp.server
