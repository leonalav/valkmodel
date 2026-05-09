from __future__ import annotations

from typing import Any

from transformers import AutoTokenizer


def load_tokenizer(tokenizer_name_or_path: str, use_fast: bool = True, **kwargs: Any):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=use_fast, **kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_tokenizer_metadata(tokenizer: Any) -> dict[str, int]:
    vocab_size = len(tokenizer) if hasattr(tokenizer, '__len__') else tokenizer.vocab_size
    return {
        'vocab_size': vocab_size,
        'bos_token_id': tokenizer.bos_token_id,
        'eos_token_id': tokenizer.eos_token_id,
        'pad_token_id': tokenizer.pad_token_id,
    }
