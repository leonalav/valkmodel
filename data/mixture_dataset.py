from __future__ import annotations

import random
from collections.abc import Iterator, Mapping
from typing import Any

from torch.utils.data import IterableDataset

from .dataset_registry import DatasetSpec
from .streaming_dataset import iter_tokenized_documents, load_stream_rows
from .streaming_mixture import StreamingMixture
from .token_stream_builder import TokenStreamBuilder


class StreamingMixtureDataset(IterableDataset):
    def __init__(
        self,
        mixture: StreamingMixture,
        tokenizer: Any,
        block_size: int,
        document_streams: Mapping[str, Iterator[str]] | None = None,
        random_seed: int = 0,
    ):
        super().__init__()
        self.mixture = mixture
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.document_streams = dict(document_streams or {})
        self.random_seed = random_seed

    def _token_iterator_for_spec(self, spec: DatasetSpec) -> Iterator[list[int]]:
        if spec.name in self.document_streams:
            for document in self.document_streams[spec.name]:
                tokens = self.tokenizer.encode(document, add_special_tokens=False)
                if tokens:
                    yield list(tokens)
            return
        rows = load_stream_rows(spec)
        yield from iter_tokenized_documents(rows, spec, self.tokenizer)

    def __iter__(self):
        rng = random.Random(self.random_seed)
        entries = list(self.mixture.entries)
        weights = [entry.weight for entry in entries]
        specs = {spec.name: spec for spec in self.mixture.specs()}
        iterators = {name: self._token_iterator_for_spec(specs[name]) for name in specs}
        stream_builder = TokenStreamBuilder(self.tokenizer.eos_token_id)

        while True:
            chosen = rng.choices(entries, weights=weights, k=1)[0]
            iterator = iterators[chosen.name]
            try:
                tokens = next(iterator)
            except StopIteration:
                del iterators[chosen.name]
                entries = [entry for entry in entries if entry.name != chosen.name]
                weights = [entry.weight for entry in entries]
                if not entries:
                    break
                continue
            stream_builder.add_document(tokens)
            for block in stream_builder.iter_blocks(self.block_size):
                yield {
                    'input_ids': block,
                    'labels': list(block),
                    'attention_mask': [1] * len(block),
                }
