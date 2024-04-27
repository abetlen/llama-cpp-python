import llama_cpp
import llama_cpp.llama_tokenizer

import gradio as gr

llama = llama_cpp.Llama.from_pretrained(
    repo_id="Qwen/Qwen1.5-0.5B-Chat-GGUF",
    filename="*q8_0.gguf",
    tokenizer=llama_cpp.llama_tokenizer.LlamaHFTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B"),
    verbose=False
)

model = "gpt-3.5-turbo"

def predict(message, history):
    messages = []

    for user_message, assistant_message in history:
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": assistant_message})
    
    messages.append({"role": "user", "content": message})

    response = llama.create_chat_completion_openai_v1(
        model=model,
        messages=messages,
        stream=True
    )

    text = ""
    for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            text += content
            yield text


js = """function () {
  gradioURL = window.location.href
  if (!gradioURL.endsWith('?__theme=dark')) {
    window.location.replace(gradioURL + '?__theme=dark');
  }
}"""

css = """
footer {
    visibility: hidden;
}
full-height {
    height: 100%;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), js=js, css=css, fill_height=True) as demo:
    gr.ChatInterface(predict, fill_height=True, examples=["What is the capital of France?", "Who was the first person on the moon?"])


if __name__ == "__main__":
    demo.launch()
