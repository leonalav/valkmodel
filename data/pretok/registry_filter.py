from __future__ import annotations

from dataclasses import replace

from data.dataset_registry import DatasetRegistry, DatasetSpec

FIELD_OVERRIDES: dict[str, str] = {"the_stack_v2": "content"}
PRETOK_EXCLUDED_DATASETS = frozenset({"creative_writing", "scientific_long"})


def build_pretok_registry(registry: DatasetRegistry) -> DatasetRegistry:
    specs: dict[str, DatasetSpec] = {}
    for name in registry.names():
        if name in PRETOK_EXCLUDED_DATASETS:
            continue
        spec = registry.get(name)
        text_field = FIELD_OVERRIDES.get(name, spec.text_field)
        if text_field != spec.text_field:
            spec = replace(spec, text_field=text_field)
        specs[name] = spec
    return DatasetRegistry(specs)
