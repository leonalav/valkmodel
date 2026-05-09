from __future__ import annotations

from dataclasses import dataclass

from .dataset_registry import DatasetRegistry, DatasetSpec


@dataclass
class MixtureEntry:
    name: str
    weight: float
    token_budget: int | None = None


class StreamingMixture:
    def __init__(self, registry: DatasetRegistry, entries: list[MixtureEntry]):
        self.registry = registry
        self.entries = entries
        self._validate()

    def _validate(self) -> None:
        total = sum(entry.weight for entry in self.entries)
        if total <= 0:
            raise ValueError('Mixture weights must sum to a positive value.')
        for entry in self.entries:
            if entry.name not in self.registry.as_dict():
                raise ValueError(f'Unknown dataset entry: {entry.name}')
            if entry.weight <= 0:
                raise ValueError('Each mixture weight must be positive.')

    def normalized_weights(self) -> dict[str, float]:
        total = sum(entry.weight for entry in self.entries)
        return {entry.name: entry.weight / total for entry in self.entries}

    def specs(self) -> list[DatasetSpec]:
        return [self.registry.get(entry.name) for entry in self.entries]

    def token_budget_by_name(self) -> dict[str, int | None]:
        return {entry.name: entry.token_budget for entry in self.entries}
