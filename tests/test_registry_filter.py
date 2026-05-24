from __future__ import annotations

from data.dataset_registry import DatasetRegistry
from data.pretok.registry_filter import (
    FIELD_OVERRIDES,
    PRETOK_EXCLUDED_DATASETS,
    build_pretok_registry,
)


def test_the_stack_v2_uses_content_field_override():
    registry = build_pretok_registry(DatasetRegistry())

    assert FIELD_OVERRIDES["the_stack_v2"] == "content"
    assert registry.get("the_stack_v2").text_field == "content"


def test_pretok_registry_excludes_unusable_datasets():
    registry = build_pretok_registry(DatasetRegistry())

    assert PRETOK_EXCLUDED_DATASETS == frozenset({"creative_writing", "scientific_long"})
    assert "creative_writing" not in registry.names()
    assert "scientific_long" not in registry.names()


def test_pretok_registry_preserves_other_datasets():
    registry = build_pretok_registry(DatasetRegistry())

    assert "fineweb_edu" in registry.names()
    assert "open_web_math" in registry.names()


def test_build_pretok_registry_does_not_mutate_source_registry():
    source = DatasetRegistry()

    _ = build_pretok_registry(source)

    assert source.get("the_stack_v2").text_field == "text"
    assert "creative_writing" in source.names()
    assert "scientific_long" in source.names()
