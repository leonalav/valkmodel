from __future__ import annotations

from pathlib import Path

import yaml

from .dataset_registry import DatasetRegistry
from .streaming_mixture import MixtureEntry, StreamingMixture


def load_mixture_config(path: str | Path, registry: DatasetRegistry | None = None) -> tuple[int | None, StreamingMixture]:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding='utf-8')) or {}
    mixture_payload = payload.get('mixture', {})
    entries = [MixtureEntry(name=name, weight=float(weight)) for name, weight in mixture_payload.items()]
    return payload.get('total_tokens'), StreamingMixture(registry or DatasetRegistry(), entries)
