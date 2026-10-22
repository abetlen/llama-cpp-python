import json
import argparse

from llama_cpp import Llama

parser = argparse.ArgumentParser(
    description="Run inference using Llama as a context manager."
)
parser.add_argument("-m", "--model", type=str, required=True, help="Path to GGUF model")
parser.add_argument(
    "-p",
    "--prompt",
    type=str,
    default="Question: What is the speed of light? Answer:",
    help="Prompt for completion",
)
args = parser.parse_args()

# Using Llama in a context manager automatically closes the model and frees resources
with Llama(model_path=args.model, verbose=False) as llm:
    output = llm(
        args.prompt,
        max_tokens=64,
        stop=["\n", "Question:"],
        echo=True,
    )
    print(json.dumps(output, indent=2))
