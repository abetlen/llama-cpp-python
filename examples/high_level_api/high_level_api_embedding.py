import argparse

from llama_cpp import Llama

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default=".//models/...")
args = parser.parse_args()

llm = Llama(model_path=args.model, embedding=True)

print(llm.create_embedding("Hello world!"))
