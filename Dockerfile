# build stage
FROM ubuntu:22.04 AS builder

# Install the package
RUN apt-get update && apt-get install -y libopenblas-dev ninja-build build-essential python3
RUN python -m pip install --upgrade pip pytest cmake scikit-build setuptools fastapi uvicorn sse-starlette

COPY . .

# Install llama_cpp_python
RUN LLAMA_OPENBLAS=1 python3 setup.py install

# final stage
FROM builder

# We need to set the host to 0.0.0.0 to allow outside access
ENV HOST 0.0.0.0

# Run the server
CMD ["python3" "-m" "llama_cpp.server"]
#CMD python3 -m llama_cpp.server
