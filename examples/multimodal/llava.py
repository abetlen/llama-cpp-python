import ctypes
import argparse
import os
import array
import sys

from llama_cpp import Llama
from llama_cpp.llava_cpp import (
    clip_model_load,
    llava_image_embed_make_with_filename,
    llava_image_embed_make_with_bytes,
    llava_image_embed_free,
    llava_validate_embed_size,
    llava_eval_image_embed,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "-m", "--model", type=str, default="../models/llava-v1.5-7b/ggml-model-q5_k.gguf"
)
parser.add_argument("--mmproj", type=str, default="llava-v1.5-7b/mmproj-model-f16.gguf")
parser.add_argument("-t", "--temp", type=float, default=0.1)
parser.add_argument(
    "-p", "--prompt", type=str, default="Describe this image in detail."
)
args = parser.parse_args()

print(f"loading clip model from {args.mmproj}")
if not os.path.exists(args.mmproj):
    raise FileNotFoundError(args.mmproj)
ctx_clip = clip_model_load(fname=args.mmproj.encode("utf-8"), verbosity=0)

image_path = os.path.join(os.path.dirname(__file__), "overfitting_lc.png")
if not os.path.exists(image_path):
    raise FileNotFoundError(image_path)
image_embed = llava_image_embed_make_with_filename(
    ctx_clip=ctx_clip, n_threads=1, image_path=image_path.encode("utf8")
)


def load_image_embed_from_file_bytes(image_path: str):
    with open(image_path, "rb") as file:
        image_bytes = file.read()
        bytes_length = len(image_bytes)
        data_array = array.array("B", image_bytes)
        c_ubyte_ptr = (ctypes.c_ubyte * len(data_array)).from_buffer(data_array)
        return llava_image_embed_make_with_bytes(
            ctx_clip=ctx_clip,
            n_threads=1,
            image_bytes=c_ubyte_ptr,
            image_bytes_length=bytes_length,
        )


print(f"loading llm model from {args.model}")
if not os.path.exists(args.model):
    raise FileNotFoundError(args.model)
llm = Llama(
    model_path=args.model, n_ctx=2048, n_gpu_layers=1
)  # longer context needed for image embeds

if not llava_validate_embed_size(llm.ctx, ctx_clip):
    raise RuntimeError("llm and mmproj model embed size mismatch")

# eval system prompt
system_prompt = "A chat between a curious human and an artificial intelligence assistant.  The assistant gives helpful, detailed, and polite answers to the human's questions.\n"
llm.eval(llm.tokenize(system_prompt.encode("utf8")))
llm.eval(llm.tokenize("\nUSER: ".encode("utf8")))

# eval image embed
n_past = ctypes.c_int(llm.n_tokens)
n_past_p = ctypes.pointer(n_past)
llava_eval_image_embed(llm.ctx, image_embed, llm.n_batch, n_past_p)
llm.n_tokens = n_past.value
llava_image_embed_free(image_embed)

# eval prompt
prompt = "Describe the visual content of this image"
llm.eval(llm.tokenize(prompt.encode("utf8")))
llm.eval(llm.tokenize("\nASSISTANT:".encode("utf8")))

# get output
print("\n")
max_target_len = 256
for i in range(max_target_len):
    t_id = llm.sample(temp=0.1)
    t = llm.detokenize([t_id]).decode("utf8")
    if t == "</s>":
        break
    print(t, end="")
    sys.stdout.flush()
    llm.eval([t_id])

print("\n")
print("done")
