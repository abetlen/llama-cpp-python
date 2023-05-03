import json
from pprint import pprint
import asyncio

import aiohttp
import llama_cpp

import llama_cpp.client as llama_cpp_client
import llama_cpp.client.aio as llama_cpp_client_aio
import llama_cpp.client.models as llama_cpp_models

LLAMA_SERVER_BASE_URL = "http://localhost:8000"


async def main():
    # Create client
    client = llama_cpp_client.LlamaCppPythonAPI(
        endpoint=LLAMA_SERVER_BASE_URL,
    )

    # List models
    models = client.get.models_v1_models_get()
    print("Models:")
    pprint(models.as_dict())

    # Completion (async)
    async with llama_cpp_client_aio.LlamaCppPythonAPI(
        endpoint=LLAMA_SERVER_BASE_URL,
    ) as aio_client:
        completion = await aio_client.create.completion_v1_completions_post(
            body={
                "model": "example",
                "prompt": "The quick brown fox jumps over the lazy dog.",
                "max_tokens": 50,
            }
        )
        print("Completion:")
        pprint(completion.as_dict())

        # Embedding (async)
        # This time we'll use a model for the request instead of an untyped dictionary
        embedding = await aio_client.create.embedding_v1_embeddings_post(
            body=llama_cpp_models.CreateEmbeddingRequest(
                model="example",
                input="The quick brown fox jumps over the lazy dog.",
            )
        )
        print("Embedding:")
        pprint(embedding.as_dict())

        # Chat completion (async)
        chat_completion = (
            await aio_client.create.chat_completion_v1_chat_completions_post(
                body=llama_cpp_models.CreateChatCompletionRequest(
                    model="example",
                    messages=[
                        llama_cpp_models.ChatCompletionRequestMessage(
                            role="system", content="You are a helpful assistant."
                        ),
                        llama_cpp_models.ChatCompletionRequestMessage(
                            role="user", content="What is the capital of France?"
                        ),
                    ],
                    temperature=0.5,
                )
            )
        )

        print("Chat completion:")
        pprint(chat_completion.as_dict())

    # Chat completion (streaming, currently can't use client)
    async with aiohttp.ClientSession() as session:
        body = llama_cpp_models.CreateChatCompletionRequest(
            model="example",
            messages=[
                llama_cpp_models.ChatCompletionRequestMessage(
                    role="system", content="You are a helpful assistant."
                ),
                llama_cpp_models.ChatCompletionRequestMessage(
                    role="user",
                    content="Tell me the story of the three little pigs in the style of a pirate.",
                ),
            ],
            max_tokens=200,
            temperature=2,
            stream=True,
        )
        async with session.post(
            f"{LLAMA_SERVER_BASE_URL}/v1/chat/completions", json=body.serialize()
        ) as response:
            async for line in response.content:
                # This sure seems like the wrong way to do this...
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    chunk_str = line[len("data: ") :].strip()
                    if chunk_str == "":
                        continue
                    elif chunk_str == "[DONE]":
                        print("")
                        break
                    else:
                        chunk_json = json.loads(chunk_str)
                        chunk = llama_cpp.ChatCompletionChunk(**chunk_json)
                        delta = chunk["choices"][0]["delta"]
                        if "content" in delta:
                            print(delta["content"], end="")

    # Completion (streaming, currently can't use client)
    async with aiohttp.ClientSession() as session:
        body = llama_cpp_models.CreateCompletionRequest(
            model="example",
            prompt="""### Human: recipe for stuffed mushrooms.
### Assistant:""",
            max_tokens=1500,
            temperature=1,
            top_p=0.55,
            top_k=33,
            stream=True,
        )
        async with session.post(
            f"{LLAMA_SERVER_BASE_URL}/v1/completions", json=body.serialize()
        ) as response:
            async for line in response.content:
                # This sure seems like the wrong way to do this...
                line = line.decode("utf-8")
                if line.startswith("data: {"):
                    chunk_str = line[len("data: ") :].strip()

                    if chunk_str == "":
                        continue
                    elif chunk_str == "[DONE]":
                        print("")
                        break
                    else:
                        chunk_json = json.loads(chunk_str)
                        chunk = llama_cpp.CompletionChunk(**chunk_json)
                        text = chunk["choices"][0]["text"]
                        print(text, end="")
    print("done!")


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
