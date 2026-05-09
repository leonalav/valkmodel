from __future__ import annotations

import math

import torch


class EvalDataCollator:
    def __init__(self, tokenizer, pad_to_multiple_of: int | None = None):
        self.pad_token_id = tokenizer.pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        max_len = max(len(example['input_ids']) for example in examples)
        if self.pad_to_multiple_of is not None:
            max_len = int(math.ceil(max_len / self.pad_to_multiple_of) * self.pad_to_multiple_of)

        input_ids = []
        attention_mask = []
        labels = []
        for example in examples:
            ids = list(example['input_ids'])
            pad_len = max_len - len(ids)
            padded = ids + [self.pad_token_id] * pad_len
            mask = [1] * len(ids) + [0] * pad_len
            label = ids + [-100] * pad_len
            input_ids.append(padded)
            attention_mask.append(mask)
            labels.append(label)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
        }
