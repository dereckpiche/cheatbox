"""The grade of one episode."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Grade:
    """The score and the metrics logged alongside it."""

    score: float
    metrics: dict[str, float]
