# Define the image argument and provide a default value
ARG IMAGE=python:3-slim-bullseye

# Use the image as specified
FROM ${IMAGE}

# Re-declare the ARG after FROM
ARG IMAGE

# Update and upgrade the existing packages 
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    ninja-build \
    build-essential

RUN python3 -m pip install --upgrade pip pytest cmake scikit-build setuptools fastapi uvicorn sse-starlette pydantic-settings

# Perform the conditional installations based on the image
RUN echo "Image: ${IMAGE}" && \
    if [ "${IMAGE}" = "python:3-slim-bullseye" ] ; then \
    echo "OpenBLAS install:" && \
    apt-get install -y --no-install-recommends libopenblas-dev && \
    LLAMA_OPENBLAS=1 pip install llama-cpp-python --verbose; \
else \
    echo "CuBLAS install:" && \
    LLAMA_CUBLAS=1 pip install llama-cpp-python --verbose; \
fi

# Clean up apt cache
RUN rm -rf /var/lib/apt/lists/*

# Set a working directory for better clarity
WORKDIR /app

# Copy files to the app directory
RUN echo "Installing model...this can take some time..."
COPY ./model.bin /app/model.bin
COPY ./start_server.sh /app/start_server.sh

# Make the server start script executable
RUN chmod +x /app/start_server.sh

# Set environment variable for the host
ENV HOST=0.0.0.0

# Expose a port for the server
EXPOSE 8000

# Run the server start script
CMD ["/bin/sh", "/app/start_server.sh"]
