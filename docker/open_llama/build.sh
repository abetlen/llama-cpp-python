#!/bin/sh

MODEL="open_llama_3b"
# Get  open_llama_3b_ggml q5_1 quantization
python3 ./hug_model.py -a SlyEcho -s ${MODEL} -f "q5_1"
ls -lh *.bin

# Build the default OpenBLAS image
docker build -t $MODEL .
docker images | egrep "^(REPOSITORY|$MODEL)"

echo
echo "To start the docker container run:"
echo "docker run -t -p 8000:8000 $MODEL"
