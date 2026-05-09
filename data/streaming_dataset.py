from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from .dataset_registry import DatasetSpec
from .filters import filter_length

try:
    from datasets import load_dataset as hf_load_dataset
except Exception:  # pragma: no cover
    hf_load_dataset = None


def extract_text(row: dict[str, Any], spec: DatasetSpec) -> str | None:
    if spec.language is not None:
        row_language = row.get('language') or row.get('lang')
        if row_language != spec.language:
            return None
    text = row.get(spec.text_field)
    if text is None:
        return None
    if not isinstance(text, str):
        text = str(text)
    return text


def iter_tokenized_documents(
    rows: Iterable[dict[str, Any]],
    spec: DatasetSpec,
    tokenizer: Any,
    min_chars: int = 1,
    max_chars: int | None = None,
) -> Iterator[list[int]]:
    for row in rows:
        text = extract_text(row, spec)
        if text is None or not filter_length(text, min_chars=min_chars, max_chars=max_chars):
            continue
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if tokens:
            yield list(tokens)


def build_dataset_load_kwargs(spec: DatasetSpec) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        'path': spec.hf_path,
        'split': spec.split,
        'streaming': True,
        'trust_remote_code': True,
    }
    if spec.revision:
        kwargs['revision'] = spec.revision
    if spec.hf_path == 'uonlp/CulturaX' and spec.language is not None:
        kwargs['name'] = spec.language
    return kwargs


def load_stream_rows(spec: DatasetSpec):
    if hf_load_dataset is None:
        raise ImportError('datasets is required to load streaming rows.')
    return hf_load_dataset(**build_dataset_load_kwargs(spec))
