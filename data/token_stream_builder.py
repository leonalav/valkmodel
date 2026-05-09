from __future__ import annotations


class TokenStreamBuilder:
    def __init__(self, eos_token_id: int):
        self.eos_token_id = eos_token_id
        self.remainder: list[int] = []

    def add_document(self, tokens: list[int]) -> None:
        self.remainder.extend(tokens)
        self.remainder.append(self.eos_token_id)

    def iter_blocks(self, block_size: int):
        while len(self.remainder) >= block_size:
            block = self.remainder[:block_size]
            self.remainder = self.remainder[block_size:]
            yield block
