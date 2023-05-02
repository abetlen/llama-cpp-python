FROM python:3-bullseye

# We need to set the host to 0.0.0.0 to allow outside access
ENV HOST 0.0.0.0

COPY . .

# Install the package
RUN apt update && apt install -y libopenblas-dev
RUN python -m pip install --upgrade pip pytest cmake scikit-build setuptools fastapi uvicorn sse-starlette

RUN LLAMA_OPENBLAS=1 python3 setup.py develop

# Run the server
CMD python3 -m llama_cpp.server