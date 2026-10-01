"""Any reasoning-gym dataset as a source. The entry is stored whole, so the
dataset's own `score_answer` grades the answer.
"""

from __future__ import annotations

import reasoning_gym

from cheatbox.sources import Problem


class ReasoningGym:
    """`ReasoningGym("sokoban", min_w=6, ...)`: the config goes to the
    dataset; puzzles are a pure function of the seed and their index.
    """

    def __init__(self, dataset: str, **config):
        """Remember the dataset name and config; one instance scores."""
        self.dataset = dataset
        self.config = config
        self.scorer = reasoning_gym.create_dataset(
            dataset, seed=0, size=1, **config
        )

    def problems(self, n: int, seed: int) -> list[Problem]:
        """The first `n` problems of the seed."""
        dataset = reasoning_gym.create_dataset(
            self.dataset, seed=seed, size=n, **self.config
        )
        return [self.problem(dataset[i]) for i in range(n)]

    @staticmethod
    def problem(entry: dict) -> Problem:
        """A reasoning-gym entry as a Problem; the entry is the spec."""
        entry = {k: entry[k] for k in ("question", "answer", "metadata")}
        return Problem(entry["question"], str(entry["answer"]), entry)

    def score(self, spec: dict, answer: str) -> float:
        """The dataset's own scoring of the answer against its entry."""
        return float(self.scorer.score_answer(answer=answer, entry=spec))
