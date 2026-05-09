from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    hf_path: str
    split: str = 'train'
    text_field: str = 'text'
    language: str | None = None
    weight: float = 1.0
    revision: str | None = None


DEFAULT_DATASET_SPECS: dict[str, DatasetSpec] = {
    'fineweb_edu': DatasetSpec(name='fineweb_edu', hf_path='HuggingFaceFW/fineweb-edu', weight=0.35),
    'the_stack_v2': DatasetSpec(name='the_stack_v2', hf_path='bigcode/the-stack-v2', weight=0.22),
    'open_web_math': DatasetSpec(name='open_web_math', hf_path='open-web-math/open-web-math', weight=0.12),
    'culturax_zh': DatasetSpec(name='culturax_zh', hf_path='uonlp/CulturaX', language='zh', weight=0.07),
    'culturax_en': DatasetSpec(name='culturax_en', hf_path='uonlp/CulturaX', language='en', weight=0.035),
    'culturax_fr': DatasetSpec(name='culturax_fr', hf_path='uonlp/CulturaX', language='fr', weight=0.025),
    'culturax_ru': DatasetSpec(name='culturax_ru', hf_path='uonlp/CulturaX', language='ru', weight=0.025),
    'culturax_vi': DatasetSpec(name='culturax_vi', hf_path='uonlp/CulturaX', language='vi', weight=0.025),
    'creative_writing': DatasetSpec(name='creative_writing', hf_path='pg19', revision='refs/convert/parquet', weight=0.08),
    'scientific_long': DatasetSpec(name='scientific_long', hf_path='scientific_papers', revision='refs/convert/parquet', weight=0.05),
}


class DatasetRegistry:
    def __init__(self, specs: dict[str, DatasetSpec] | None = None):
        self._specs = dict(specs or DEFAULT_DATASET_SPECS)

    def get(self, name: str) -> DatasetSpec:
        return self._specs[name]

    def names(self) -> list[str]:
        return list(self._specs.keys())

    def as_dict(self) -> dict[str, DatasetSpec]:
        return dict(self._specs)
