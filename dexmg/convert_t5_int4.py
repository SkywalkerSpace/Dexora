"""Load a T5 encoder in 4-bit and save the quantized model in-place.

Example:
    python scripts/convert_t5_int4.py /path/to/google/t5-v1_1-xxl
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, T5EncoderModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quantize a T5 encoder to NF4 int4 and save it to the same directory."
    )
    parser.add_argument(
        "model",
        type=Path,
        help="Existing local model directory to quantize in-place.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--model-max-length",
        type=int,
        default=120,
        help="Maximum tokenizer sequence length to save (default: 120).",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not access the Hugging Face Hub while loading.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.model.is_dir():
        raise ValueError(
            f"model must be an existing local directory to save in-place: {args.model}"
        )

    model_path = str(args.model)
    output_dir = args.model
    compute_dtype = torch.bfloat16

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    load_kwargs = {
        "cache_dir": args.cache_dir,
        "local_files_only": args.local_files_only,
        "low_cpu_mem_usage": True,
        "torch_dtype": compute_dtype,
        "quantization_config": quantization_config,
        "device_map": {"": 0},
    }

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        model_max_length=args.model_max_length,
    )
    model = T5EncoderModel.from_pretrained(model_path, **load_kwargs).eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved int4 T5 encoder and tokenizer to {output_dir}")


if __name__ == "__main__":
    main()