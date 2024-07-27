import os
import argparse
import llama_cpp


def main(args):
    fname_inp = args.fname_inp.encode("utf-8")
    fname_out = args.fname_out.encode("utf-8")
    if not os.path.exists(fname_inp):
        raise RuntimeError(f"Input file does not exist ({fname_inp})")
    if os.path.exists(fname_out):
        raise RuntimeError(f"Output file already exists ({fname_out})")
    ftype = args.type
    args = llama_cpp.llama_model_quantize_default_params()
    args.ftype = ftype
    return_code = llama_cpp.llama_model_quantize(fname_inp, fname_out, args)
    if return_code != 0:
        raise RuntimeError("Failed to quantize model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fname_inp", type=str, help="Path to input model")
    parser.add_argument("fname_out", type=str, help="Path to output model")
    parser.add_argument(
        "type",
        type=int,
        help="Type of quantization (2: q4_0, 3: q4_1), see llama_cpp.py for enum",
    )
    args = parser.parse_args()
    main(args)
