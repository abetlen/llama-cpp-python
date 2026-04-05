"""
Optimized llama-cpp-python configuration for:
  Lenovo Legion Slim 5 (16" RH8)
  - CPU:  Intel Core i7-13700H (6P + 8E cores)
  - GPU:  NVIDIA GeForce RTX 4060 Laptop (8 GB VRAM, GDDR6)
  - RAM:  16 GB DDR5-5200
  - SSD:  1 TB NVMe

Install with CUDA support first:
  CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
"""

import argparse
import json

from llama_cpp import Llama

# ---------------------------------------------------------------------------
# Hardware constants for this machine
# ---------------------------------------------------------------------------
VRAM_GB = 8          # RTX 4060 Laptop VRAM
N_PHYSICAL_CORES = 6  # P-cores only (best single-thread perf on i7-13700H)

# ---------------------------------------------------------------------------
# Recommended quantisation levels (pick one based on your model size)
# ---------------------------------------------------------------------------
# Model 7B / 8B:
#   Q5_K_M  → ~5.5 GB VRAM  ✅ recommended
#   Q6_K    → ~6.5 GB VRAM  ✅ excellent quality
#   Q8_0    → ~8.5 GB VRAM  ⚠️  tight fit, may spill to CPU RAM
#
# Model 13B:
#   Q4_K_M  → ~7.5 GB VRAM  ✅ fits
#   Q5_K_M  → ~9.0 GB VRAM  ❌ exceeds VRAM


def build_llm(
    model_path: str,
    n_ctx: int = 4096,
    n_gpu_layers: int = -1,  # -1 = offload all layers to GPU
    n_batch: int = 512,
    verbose: bool = False,
) -> Llama:
    """
    Create a Llama instance tuned for the Legion Slim 5 / RTX 4060 laptop.

    Args:
        model_path:   Path to the .gguf model file.
        n_ctx:        Context window size (tokens).  4096 is safe for 8 GB VRAM.
        n_gpu_layers: Number of transformer layers to offload to the GPU.
                      Use -1 to offload everything (default).  Reduce if you
                      see CUDA out-of-memory errors.
        n_batch:      Batch size for prompt evaluation.
        verbose:      Print llama.cpp loading messages.

    Returns:
        A ready-to-use Llama instance.
    """
    return Llama(
        model_path=model_path,
        # --- GPU offload ---
        n_gpu_layers=n_gpu_layers,  # RTX 4060 has 8 GB – offload as much as fits
        offload_kqv=True,           # keep KV-cache on GPU for faster inference
        # --- CPU threads ---
        n_threads=N_PHYSICAL_CORES,  # use P-cores only for best throughput
        n_threads_batch=N_PHYSICAL_CORES,
        # --- Context / batching ---
        n_ctx=n_ctx,
        n_batch=n_batch,
        # --- Memory ---
        use_mmap=True,   # fast model loading from NVMe SSD
        use_mlock=False, # don't pin 16 GB RAM – OS needs headroom
        # --- Misc ---
        verbose=verbose,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run inference optimised for the Lenovo Legion Slim 5 / RTX 4060"
    )
    parser.add_argument(
        "-m", "--model",
        required=True,
        help="Path to the .gguf model file (e.g. mistral-7b-Q5_K_M.gguf)",
    )
    parser.add_argument(
        "-p", "--prompt",
        default="What are the names of the planets in the solar system?",
        help="Prompt text",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=256,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--n-ctx", type=int, default=4096,
        help="Context window size",
    )
    parser.add_argument(
        "--n-gpu-layers", type=int, default=-1,
        help="GPU layers to offload (-1 = all)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print llama.cpp loading messages",
    )
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    print(f"GPU layers   : {'all' if args.n_gpu_layers == -1 else args.n_gpu_layers}")
    print(f"Context size : {args.n_ctx} tokens\n")

    llm = build_llm(
        model_path=args.model,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        verbose=args.verbose,
    )

    output = llm(
        args.prompt,
        max_tokens=args.max_tokens,
        stop=["Q:", "\n\n"],
        echo=True,
    )

    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
