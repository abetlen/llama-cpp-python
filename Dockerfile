FROM python:3-buster

# We need to set the host to 0.0.0.0 to allow outside access
ENV HOST 0.0.0.0

# Install the package
RUN pip install llama-cpp-python[server]

# Run the server
CMD python3 -m llama_cpp.server