from __future__ import annotations

from data.dataset_registry import DatasetRegistry


def test_fineweb_edu_defaults_to_sample_10bt_subset():
    registry = DatasetRegistry()

    assert registry.get("fineweb_edu").subset == "sample-10BT"
