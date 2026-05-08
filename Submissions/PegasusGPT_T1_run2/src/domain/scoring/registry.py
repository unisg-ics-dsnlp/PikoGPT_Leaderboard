from __future__ import annotations

from domain.scoring.base import BenchmarkAdapter
from domain.scoring.benchmarks.hellaswag import HellaSwagAdapter
from domain.scoring.benchmarks.openbookqa import OpenBookQAAdapter
from domain.scoring.benchmarks.winogrande import WinoGrandeAdapter


class BenchmarkRegistry:
    def __init__(self) -> None:
        self._adapters: list[BenchmarkAdapter] = []

    def register(self, adapter: BenchmarkAdapter) -> None:
        self._adapters.append(adapter)

    def detect(self, prompt: str) -> BenchmarkAdapter | None:
        for adapter in self._adapters:
            if adapter.detect(prompt):
                return adapter
        return None


def default_registry() -> BenchmarkRegistry:
    registry = BenchmarkRegistry()
    registry.register(HellaSwagAdapter())
    registry.register(OpenBookQAAdapter())
    registry.register(WinoGrandeAdapter())
    return registry
