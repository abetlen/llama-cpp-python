FROM python:3-bullseye

# We need to set the host to 0.0.0.0 to allow outside access
ENV HOST 0.0.0.0

# Install the package
RUN apt update && apt install -y libopenblas-dev && LLAMA_OPENBLAS=1 pip install llama-cpp-python[server]

# Run the server
CMD python3 -m llama_cpp.server