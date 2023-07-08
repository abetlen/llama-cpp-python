#!/bin/sh

MODEL="open_llama_3b"

# Start Docker container
docker run --cap-add SYS_RESOURCE -p 8000:8000 -t $MODEL &
sleep 10
echo
docker ps | egrep "(^CONTAINER|$MODEL)"

# Test the model works
echo
curl -X 'POST'   'http://localhost:8000/v1/completions'   -H 'accept: application/json'   -H 'Content-Type: application/json'   -d '{
  "prompt": "\n\n### Instructions:\nWhat is the capital of France?\n\n### Response:\n",
  "stop": [
    "\n",
    "###"
  ]
}' | grep Paris
if [ $? -eq 0 ]
then
    echo
    echo "$MODEL is working!!"
else
    echo
    echo "ERROR: $MODEL not replying."
    exit 1
fi
