#!/usr/bin/env python3
"""
Build an 8292-vocab identity tokenizer for the streaming VQGAN codes.

Design:
- Code tokens: v0000..v8191 -> ids 0..8191 (identity mapping for VQGAN codes)
- Template tokens: 'User:' -> 8192, 'Assistant:' -> 8193
- Special tokens: '<pad>' -> 8194, '<bos>' -> 8195, '<eos>' -> 8196
- Fillers: <extra_000> .. as needed to reach vocab_size=8292

The produced files are compatible with AutoTokenizer.from_pretrained():
- tokenizer.json (WordLevel + WhitespaceSplit)
- tokenizer_config.json (includes chat_template for llama2-lite)
- special_tokens_map.json

Usage:
  python3 streaming_video/tools/build_identity_tokenizer.py \
    --out-dir streaming_video/saved_models/lvm-llama2-7b
"""

import argparse
import json
import os

VOCAB_SIZE = 8292
NUM_CODE_TOKENS = 8192  # v0000..v8191


def build_vocab():
    vocab = {}
    # 0..8191 code tokens
    for i in range(NUM_CODE_TOKENS):
        vocab[f"v{i:04d}"] = i

    # reserved/template/specials
    current = NUM_CODE_TOKENS
    reserve = {
        "User:": current,
        "Assistant:": current + 1,
        "<pad>": current + 2,
        "<bos>": current + 3,
        "<eos>": current + 4,
    }
    vocab.update(reserve)
    current += len(reserve)

    # Fillers to reach exact vocab size
    while len(vocab) < VOCAB_SIZE:
        name = f"<extra_{len(vocab) - NUM_CODE_TOKENS - len(reserve):03d}>"
        if name in vocab:
            name = f"<extra_{len(vocab):03d}>"
        vocab[name] = current
        current += 1

    assert len(vocab) == VOCAB_SIZE
    # inverse check: ids cover [0, VOCAB_SIZE)
    ids = sorted(vocab.values())
    assert ids[0] == 0 and ids[-1] == VOCAB_SIZE - 1 and ids == list(range(VOCAB_SIZE))
    return vocab


def write_tokenizer_json(path: str, vocab: dict):
    # Added tokens: specials only (keep User:/Assistant: as normal tokens)
    specials = []
    for tok in ("<pad>", "<bos>", "<eos>"):
        specials.append(
            {
                "id": vocab[tok],
                "content": tok,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": True,
                "special": True,
            }
        )

    tok = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": specials,
        "normalizer": None,
        "pre_tokenizer": {"type": "WhitespaceSplit"},
        "post_processor": None,
        "decoder": None,
        # WordLevel requires a string unk_token in the serialized schema.
        # Ensure '<unk>' exists in the vocab; if not, reserve one.
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<unk>"},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tok, f, ensure_ascii=False, indent=2)


def write_tokenizer_config_json(path: str):
    # Minimal chat_template that pairs with TEMPLATE_REGISTRY["llama2-lite"]
    chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}User: {{ message['content'] }}\n"
        "{% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}\n"
        "{% elif message['role'] == 'system' and message['content'] %}{{ message['content'] }}\n"
        "{% endif %}{% endfor %}{% if add_generation_prompt %}Assistant:{% endif %}"
    )
    cfg = {
        "clean_up_tokenization_spaces": False,
        "model_max_length": 4096,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "chat_template": chat_template,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def write_special_tokens_map_json(path: str):
    # Map commonly used specials; ids are derived from tokenizer.json
    m = {
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "pad_token": "<pad>",
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    vocab = build_vocab()
    # Ensure '<unk>' exists in vocab; if not, repurpose the first extra token
    if "<unk>" not in vocab:
        # Find an id not used yet within [0, VOCAB_SIZE)
        available_ids = set(range(VOCAB_SIZE)) - set(vocab.values())
        if available_ids:
            new_id = min(available_ids)
        else:
            # Replace the last extra token deterministically
            # Find a candidate extra token name
            extras = [k for k in vocab.keys() if k.startswith("<extra_")]
            if not extras:
                # As a fallback, replace 'Assistant:' which we can re-add
                victim = "Assistant:"
            else:
                victim = sorted(extras)[-1]
            new_id = vocab[victim]
            del vocab[victim]
        vocab["<unk>"] = new_id
    write_tokenizer_json(os.path.join(args.out_dir, "tokenizer.json"), vocab)
    write_tokenizer_config_json(os.path.join(args.out_dir, "tokenizer_config.json"))
    write_special_tokens_map_json(os.path.join(args.out_dir, "special_tokens_map.json"))

    print(f"Identity tokenizer written to {args.out_dir}")


if __name__ == "__main__":
    main()
