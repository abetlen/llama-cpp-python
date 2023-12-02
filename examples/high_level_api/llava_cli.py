import base64
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler

llava_folder = "path/to/folder/llava/llava_v1.5_7b/"
file_path = 'path/to/your/image.png'

clip_model_path = f"{llava_folder}mmproj-model-f16.gguf"
model_path = f"{llava_folder}llava_v1.5_7b_q4_k.gguf"

def image_to_base64_data_uri(file_path):
    with open(file_path, "rb") as img_file:
        base64_data = base64.b64encode(img_file.read()).decode('utf-8')
        return f"data:image/png;base64,{base64_data}"


print('load image')
data_uri = image_to_base64_data_uri(file_path)
print('image encoded')

print("init image handler")
chat_handler = Llava15ChatHandler(
    clip_model_path=clip_model_path,
    verbose=True)

llm = Llama(
    model_path=model_path,
    chat_format="llava-1-5",
    chat_handler=chat_handler,
    n_ctx=4096,# n_ctx should be increased to accomodate the image embedding
    logits_all=True,# needed to make llava work
)

messages=[
            {"role": "system", "content": "You are an assistant who perfectly describes images."},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": "Describe this image in detail please."}
                ]
            }
        ]
print('init chat completion')
# Call create_chat_completion to get the response
response = llm.create_chat_completion(messages=messages, )
print(response)
print()
print()
print(response['choices'][0]['message']['content'])
