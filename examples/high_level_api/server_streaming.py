import requests
import json

url = "http://localhost:8000/v1/chat/completions"
headers = {"accept": "application/json", "Content-Type": "application/json"}

stream = True
data = {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write me quicksort in Python."},
    ],
    "max_tokens": 200,
    "stream": stream,
}

response = requests.post(url, headers=headers, json=data, stream=True)
if not stream:
    print(response.json()["choices"][0]["message"]["content"])
else:
    flag = 0
    string = []
    for chunk in response:
        string.append(chunk.decode("utf-8"))
        flag += 1
        if flag % 3 == 0:
            if string[0].startswith("data:"):
                delta = json.loads("".join(string)[5:])["choices"][0]["delta"]
                if delta.get("content", ""):
                    print(delta["content"], end="", flush=True)
            string = []
