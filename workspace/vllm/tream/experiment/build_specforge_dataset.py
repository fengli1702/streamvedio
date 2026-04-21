#!/usr/bin/env python
"""
Construct SpecForge-compatible streaming datasets from raw VQGAN frames.

Each sample follows the conversation schema expected by SpecForge online
training:

    {
        "id": "...",
        "conversations": [
            {"role": "user", "content": "<prompt_text>"},
            {"role": "assistant", "content": "<target_text>"}
        ]
    }

The prompt spans `context_length` frames and the target spans the following
`inference_length` frames. Frames are first encoded with the streaming VQGAN
encoder so that the resulting text round-trips to the exact same token IDs
under the custom LLaMA tokenizer shipped with the streaming model.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoTokenizer

from lvm_tokenizer.muse import VQGANModel
from lvm_tokenizer.utils import ENCODING_SIZE, RAW_VQGAN_PATH

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass
class SequenceInfo:
    name: str
    path: Path
    frames: list[Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SpecForge conversational dataset from streaming video frames.")
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Directory containing frame folders (recursively scanned).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write train.jsonl / val.jsonl.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=Path("./saved_models/lvm-llama2-7b"),
        help="Path to the target LLaMA tokenizer.",
    )
    parser.add_argument(
        "--vqgan-path",
        type=Path,
        default=Path(RAW_VQGAN_PATH),
        help="Path to the pretrained streaming VQGAN checkpoints.",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=4,
        help="Number of past frames (C) to include in the prompt.",
    )
    parser.add_argument(
        "--inference-length",
        type=int,
        default=4,
        help="Number of future frames (L) to use as the supervision target.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Ratio of samples written to train.jsonl (rest go to val.jsonl).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Sliding window stride (in frames) between successive samples.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for VQGAN encoding.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device string (defaults to 'cuda' if available, else 'cpu').",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on the number of frames to load per sequence.",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="Optional cap on the number of sequences to process.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing train/val files if they already exist.",
    )
    parser.add_argument(
        "--verify-roundtrip",
        type=int,
        default=1,
        help="Number of samples to verify prompt/target round-trip tokenisation (0 to disable).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for frame directories (default: only direct children).",
    )
    return parser.parse_args()


def iter_frame_directories(root: Path, recursive: bool) -> Iterator[Path]:
    """
    Yield directories that contain at least one image file. If recursive is False,
    only inspect the direct children of root. Otherwise perform a depth-first scan,
    stopping descent once a directory with frames is found.
    """
    if not root.is_dir():
        raise FileNotFoundError(f"Input root {root} is not a directory.")

    stack = [root]
    visited: set[Path] = set()

    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)

        images = list_image_files(current)
        if images:
            yield current
            continue

        if recursive:
            children = [p for p in current.iterdir() if p.is_dir()]
            children.sort(key=lambda p: p.name)
            stack.extend(reversed(children))


def list_image_files(directory: Path) -> list[Path]:
    files = [
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and not path.name.startswith(".")
    ]
    files.sort(key=lambda p: p.name)
    return files


def load_sequence_info(root: Path, recursive: bool, max_frames: int | None, max_sequences: int | None) -> list[SequenceInfo]:
    sequences: list[SequenceInfo] = []
    for seq_path in iter_frame_directories(root, recursive=recursive):
        frames = list_image_files(seq_path)
        if not frames:
            continue
        if max_frames is not None:
            frames = frames[:max_frames]
        try:
            rel = seq_path.relative_to(root)
            parts = [p for p in rel.parts if p not in {".", ""}]
        except ValueError:
            parts = []
        if not parts:
            parts = [seq_path.name]
        seq_name = "_".join(parts)
        sequences.append(SequenceInfo(name=seq_name, path=seq_path, frames=frames))
        if max_sequences is not None and len(sequences) >= max_sequences:
            break
    return sequences


def init_vqgan(path: Path, device: torch.device) -> VQGANModel:
    model = VQGANModel.from_pretrained(str(path))
    model = model.to(device)
    model.eval()
    return model


def encode_frames(
    frame_paths: Sequence[Path],
    encoder: VQGANModel,
    image_transform: transforms.Compose,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """
    Encode a set of frames with the VQGAN encoder, returning a CPU tensor with shape
    [num_frames, ENCODING_SIZE].
    """
    tokens: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(frame_paths), batch_size):
            batch_paths = frame_paths[start:start + batch_size]
            images = []
            for path in batch_paths:
                with Image.open(path) as img:
                    images.append(image_transform(img.convert("RGB")))
            image_tensor = torch.stack(images, dim=0).to(device, non_blocking=True)
            _, frame_tokens = encoder.encode(image_tensor)
            tokens.append(frame_tokens.to(dtype=torch.int32, device="cpu"))
    if not tokens:
        raise RuntimeError("No frames were encoded.")
    return torch.cat(tokens, dim=0)


def compute_num_samples(num_frames: int, context_length: int, inference_length: int, stride: int) -> int:
    span = context_length + inference_length
    if num_frames < span:
        return 0
    max_start = num_frames - span
    return max_start // stride + 1


def flatten_frame_span(tokens: torch.Tensor, start: int, length: int) -> list[int]:
    """
    Slice a [frames, ENCODING_SIZE] tensor, flatten the requested span, and return as python ints.
    """
    end = start + length
    return tokens[start:end].reshape(-1).tolist()


def ids_to_piece_string(tokenizer: AutoTokenizer, ids: Sequence[int]) -> str:
    """
    Convert ids to a textual form using the tokenizer's convert_tokens_to_string helper, which
    maintains the whitespace markers for LLaMA-style tokenizers.
    """
    pieces = tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=False)
    return tokenizer.convert_tokens_to_string(pieces)


def roundtrip_check(tokenizer: AutoTokenizer, text: str, expected_ids: Sequence[int]) -> None:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if encoded != list(expected_ids):
        raise ValueError(
            "Tokenizer round-trip mismatch:\n"
            f"  expected ids: {expected_ids[:16]}... (len={len(expected_ids)})\n"
            f"  got ids:      {encoded[:16]}... (len={len(encoded)})"
        )


def main() -> None:
    args = parse_args()

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    context_length = args.context_length
    inference_length = args.inference_length
    stride = args.stride

    if context_length <= 0 or inference_length <= 0:
        raise ValueError("context_length and inference_length must be positive.")
    if stride <= 0:
        raise ValueError("stride must be positive.")
    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("train_ratio must be between 0 and 1 (exclusive).")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    if not args.overwrite:
        for path in (train_path, val_path):
            if path.exists():
                raise FileExistsError(f"{path} already exists. Use --overwrite to replace.")

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer_path),
        trust_remote_code=True,
    )
    tokenizer.model_max_length = max(
        getattr(tokenizer, "model_max_length", 0) or 0,
        (context_length + inference_length) * ENCODING_SIZE,
    )
    image_transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ]
    )
    encoder = init_vqgan(args.vqgan_path, device=device)

    sequences = load_sequence_info(
        root=args.input_root,
        recursive=args.recursive,
        max_frames=args.max_frames,
        max_sequences=args.max_sequences,
    )

    if not sequences:
        raise ValueError(f"No frame sequences found under {args.input_root}.")

    print(f"Discovered {len(sequences)} frame sequences under {args.input_root}.")
    print(f"Writing outputs to {output_dir} (train ratio={args.train_ratio:.2f}).")

    total_train = 0
    total_val = 0
    total_samples = 0
    roundtrip_budget = args.verify_roundtrip

    with train_path.open("w", encoding="utf-8") as train_file, val_path.open("w", encoding="utf-8") as val_file:
        for seq in tqdm(sequences, desc="Sequences"):
            frame_tensor = encode_frames(
                frame_paths=seq.frames,
                encoder=encoder,
                image_transform=image_transform,
                device=device,
                batch_size=args.batch_size,
            )

            num_frames = frame_tensor.shape[0]
            samples_in_seq = compute_num_samples(
                num_frames=num_frames,
                context_length=context_length,
                inference_length=inference_length,
                stride=stride,
            )
            if samples_in_seq == 0:
                print(f"[skip] {seq.name}: not enough frames ({num_frames}) for span={context_length + inference_length}.")
                continue

            train_cutoff = min(samples_in_seq, int(math.ceil(samples_in_seq * args.train_ratio)))
            if samples_in_seq > 1 and train_cutoff == samples_in_seq:
                train_cutoff = samples_in_seq - 1
            seq_train = 0
            seq_val = 0

            for sample_idx in range(samples_in_seq):
                start_frame = sample_idx * stride
                if start_frame + context_length + inference_length > num_frames:
                    # Guard against overshooting when max_start % stride != 0.
                    start_frame = num_frames - (context_length + inference_length)
                prompt_ids = flatten_frame_span(frame_tensor, start_frame, context_length)
                target_ids = flatten_frame_span(frame_tensor, start_frame + context_length, inference_length)

                prompt_text = ids_to_piece_string(tokenizer, prompt_ids)
                target_text = ids_to_piece_string(tokenizer, target_ids)

                sample_id = f"{seq.name}_{sample_idx:06d}"

                if roundtrip_budget > 0:
                    try:
                        roundtrip_check(tokenizer, prompt_text, prompt_ids)
                        roundtrip_check(tokenizer, target_text, target_ids)
                    except ValueError as exc:
                        raise ValueError(f"Round-trip verification failed for sample {sample_id}") from exc
                    roundtrip_budget -= 1

                record = {
                    "id": sample_id,
                    "conversations": [
                        {"role": "user", "content": prompt_text},
                        {"role": "assistant", "content": target_text},
                    ],
                }

                line = json.dumps(record, ensure_ascii=True)
                if sample_idx < train_cutoff:
                    train_file.write(line + "\n")
                    total_train += 1
                    seq_train += 1
                else:
                    val_file.write(line + "\n")
                    total_val += 1
                    seq_val += 1

            total_samples += samples_in_seq
            print(f"[sequence] {seq.name}: frames={num_frames} samples={samples_in_seq} train={seq_train} val={seq_val}")

            del frame_tensor
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print("Finished dataset construction.")
    print(f"  sequences processed: {len(sequences)}")
    print(f"  total samples:       {total_samples}")
    print(f"  train samples:       {total_train}")
    print(f"  val samples:         {total_val}")
    print(f"  output files:        {train_path}, {val_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        sys.exit(130)
