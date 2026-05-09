from __future__ import annotations

import torch


class PackedDataCollator:
    def __init__(self, tokenizer, block_size: int):
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        input_ids = torch.tensor([example['input_ids'] for example in examples], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }
