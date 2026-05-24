from __future__ import annotations

from data.dataset_registry import DatasetSpec
from data.streaming_dataset import build_dataset_load_kwargs


def test_fineweb_subset_defaults_to_sample_10bt():
    spec = DatasetSpec(
        name="fineweb_edu",
        hf_path="HuggingFaceFW/fineweb-edu",
        subset="sample-10BT",
    )

    kwargs = build_dataset_load_kwargs(spec)

    assert kwargs["name"] == "sample-10BT"


def test_culturax_language_still_maps_to_name_when_no_subset():
    spec = DatasetSpec(
        name="culturax_en",
        hf_path="uonlp/CulturaX",
        language="en",
    )

    kwargs = build_dataset_load_kwargs(spec)

    assert kwargs["name"] == "en"
