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
    libopenblas-dev \
    build-essential

RUN mkdir /app
WORKDIR /app
COPY . /app

RUN python3 -m pip install --upgrade pip

RUN make deps && make build && make clean

# Set environment variable for the host
ENV HOST=0.0.0.0
ENV PORT=8000

# Expose a port for the server
EXPOSE 8000

# Run the server start script
CMD ["/bin/sh", "/app/docker/simple/run.sh"]
