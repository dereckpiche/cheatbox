"""Task sources: where problems come from and how answers are scored. A
source is named like `rg:sokoban` (a reasoning-gym dataset) or `mbpp`
(coding), and takes a config dict; `load` returns a cached instance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Protocol


@dataclass
class Problem:
    """One problem: what the agent reads, the reference answer, and what
    the scorer needs.
    """

    question: str
    answer: str
    spec: dict = field(default_factory=dict)


class Source(Protocol):
    """What a task source provides."""

    def problems(self, n: int, seed: int) -> list[Problem]:
        """At most `n` problems, reproducible from `seed`."""

    def score(self, spec: dict, answer: str) -> float:
        """The answer's score in [0, 1]; 1 means fully correct."""


def load(name: str, config: dict | None = None) -> Source:
    """The source called `name` with `config`, built once per pair."""
    return _load(name, json.dumps(config or {}, sort_keys=True))


@lru_cache(maxsize=None)
def _load(name: str, config_json: str) -> Source:
    """Build a source; the config travels as JSON so it can be a cache key."""
    config = json.loads(config_json)
    kind, _, rest = name.partition(":")
    if kind == "rg":
        from cheatbox.sources.reasoning_gym import ReasoningGym

        return ReasoningGym(rest, **config)
    if kind == "mbpp":
        from cheatbox.sources.mbpp import Mbpp

        return Mbpp(**config)
    raise ValueError(f"unknown source {name!r}; use rg:<dataset> or mbpp")
