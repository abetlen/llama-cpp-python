from llama_cpp import ChatCompletionMessage, Llama
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="<your path to your ggml-model.bin>")
args = parser.parse_args()

llm = Llama(model_path=args.model)

# Create a list of messages
messages = [
    ChatCompletionMessage(role='system', content='start chat'),
    ChatCompletionMessage(role='user', content='Hello')
]

while True:
    # Generate a response
    response = llm.create_chat_completion(
        messages,
        temperature=0.2,
        top_p=0.95,
        top_k=40,
        stream=False,
        stop=[],
        max_tokens=256,
        repeat_penalty=1.1,
    )

    output = response['choices'][0]['message']['content']
    print(f"Rob: {output}")

    # Append assistant's message to conversation history
    messages.append(ChatCompletionMessage(role='assistant', content=output))

    user_message = input("User: ")
    messages.append(ChatCompletionMessage(role='user', content=user_message))
