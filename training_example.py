"""Train an LLM against cheatbox with verl.

What verl does for us
---------------------
verl runs the RL loop: it samples tasks from a parquet file, lets vLLM
generate the model's turns, executes the model's tool calls, scores each
finished trajectory, and updates the policy with GRPO. We plug three things
into it: a dataset, a tool, and an agent loop.

The contract with verl
----------------------
verl owns the tokens. For every trajectory it calls, exactly once,

    output = await loop.run(sampling_params, **row)

on an asyncio event loop shared by hundreds of trajectories at a time, so
anything blocking must go through `asyncio.to_thread`. `row` is the
parquet row: `raw_prompt` (the chat messages) and `extra_info` (ours).
`sampling_params` is what vLLM gets (temperature, top_p, max_tokens).
Inside `run` there is one way to call the model:

    out = await self.server_manager.generate(
        request_id, prompt_ids, sampling_params
    )                                        # out.token_ids

Tokens in, tokens out, callable as many times as we like. `run` must
return an `AgentLoopOutput`:

    prompt_ids      the tokenized prompt
    response_ids    everything after it, model tokens and tool outputs
                    interleaved in order, at most rollout.response_length
    response_mask   1 where the model generated, 0 where we inserted
                    text (tool outputs); only the 1s get a gradient
    reward_score    optional; if set, verl skips its reward function
    extra_fields["reward_extra_info"]
                    optional dict of numbers; averaged into val-aux/
                    metrics during validation

That is the whole interface. `ToolAgentLoop` is verl's stock
implementation of it, a three-state machine:

    PENDING           tokenize the chat with the tool schemas
    GENERATING        one `generate` call, tokens appended with mask 1,
                      tool calls parsed out of the text
    PROCESSING_TOOLS  run each call through `BaseTool.execute`, append
                      the outputs as tool messages with mask 0
    TERMINATED        no tool call in the last turn, or response_length
                      or max_assistant_turns reached

Each state is a method; `_handle_generating_state` runs before every
model call, which is where a token budget belongs.

A tool is a `BaseTool` with a JSON schema the model sees and

    await tool.execute(instance_id, parameters, agent_data)
        -> (ToolResponse(text=...), reward: float, metrics: dict)

called once per parsed call. `parameters` is the model's argument dict.
`agent_data.tools_kwargs` is the row's `tools_kwargs` column, which the
loop may overwrite: that is how our loop tells the tool which Episode a
trajectory belongs to.

What we plug in
---------------
1. `TaskRecord`: the whole cheatbox Task (prompts, workspace files, board,
   cheat answer), serialised into `extra_info`.
2. `Bash`: the one tool. `execute` runs the command in the Episode.
3. `CheatboxLoop`: `ToolAgentLoop` with `run` wrapped to start an
   Episode first and grade it last, and `_handle_generating_state`
   wrapped to enforce the budget. The grade is `reward_score`; its
   metrics are `reward_extra_info`.
4. An episode log. verl reports per-episode numbers only at validation,
   so the loop also appends one JSON line per finished episode (its
   grade, whether it showed the black file's body, token counts and full
   transcript) to the file named by
   `CHEATBOX_EPISODE_LOG`. `progress` averages the lines that arrived
   since the previous step into `rollout_train` and `rollout_valid`,
   and sends a few of their transcripts to wandb as tables
   (`samples/rollout_train`, `samples/rollout_valid`): a random rewarded
   episode, a random unrewarded one, and one random episode.

Everything else is configuration, in `training_example_config/`:

- `train.yaml`: every training knob (model, data, batch, lengths, GPUs),
  layered on top of verl's defaults. This is what you launch.
- `tools.yaml`: the tool classes the model may call, by import path.
- `agent_loop.yaml`: the loop classes, by import path, with their
  arguments.

verl reads the last two from separate files, which is why they exist.

Usage
-----
    python training_example.py data
    python -m verl.trainer.main_ppo --config-path $PWD/training_example_config --config-name train
    ... | python training_example.py progress
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, TextIO

import pyarrow as pa
import pyarrow.parquet as pq
from verl.experimental.agent_loop.agent_loop import register
from verl.experimental.agent_loop.tool_agent_loop import (
    AgentState,
    ToolAgentLoop,
)
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

import cheatbox

# ------------------------------------------------------------------ dataset


@dataclass
class TaskRecord:
    """A cheatbox Task as it travels through verl's dataset.

    verl keeps `extra_info` as an opaque dict per row and gives it back to the
    agent loop, so we serialise the Task into it and read it back. The dicts
    become JSON so parquet stays flat.
    """

    task_id: int
    system: str
    user: str
    files: str
    source: str
    config: str
    spec: str
    cheat: str
    black: str
    budget: int = cheatbox.prompts.BUDGET_TOKENS

    @classmethod
    def from_task(cls, task: cheatbox.Task) -> TaskRecord:
        """Serialise a Task."""
        return cls(
            task.id,
            task.system,
            task.user,
            json.dumps(task.files),
            task.source,
            json.dumps(task.config),
            json.dumps(task.spec),
            task.cheat,
            task.black,
            task.budget,
        )

    def to_task(self) -> cheatbox.Task:
        """Rebuild the Task from a row's extra_info."""
        return cheatbox.Task(
            int(self.task_id),
            self.system,
            self.user,
            json.loads(self.files),
            self.source,
            json.loads(self.config),
            json.loads(self.spec),
            self.cheat,
            self.black,
            int(self.budget),
        )

    def to_row(self, index: int, split: str) -> dict:
        """The parquet row verl expects: chat prompt plus our extra_info.
        `split` rides along so the loop can tag its episode records.
        """
        messages = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user},
        ]
        return {
            "data_source": self.source,
            "prompt": messages,
            "ability": "agent",
            "reward_model": {"style": "rule", "ground_truth": ""},
            "extra_info": {"index": index, "split": split, **asdict(self)},
        }


def write_dataset(
    out: Path,
    source: str,
    config: dict,
    n: int,
    budget: int,
    seed: int,
    cheat: bool,
) -> None:
    """Write train.parquet and val.parquet: the same `n` problems, the val
    copies carrying decoy cheat files, the train copies too if `cheat` is
    off.
    """
    out.mkdir(parents=True, exist_ok=True)
    splits = cheatbox.tasks.generate(source, n, budget, config, seed, cheat)
    for split, tasks in splits.items():
        records = [TaskRecord.from_task(t) for t in tasks]
        rows = [r.to_row(i, split) for i, r in enumerate(records)]
        pq.write_table(pa.Table.from_pylist(rows), out / f"{split}.parquet")
        print(f"{split}: {len(rows)} tasks -> {out / split}.parquet")


# --------------------------------------------------------------------- tool

BASH_SCHEMA = OpenAIFunctionToolSchema.model_validate(
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a bash command in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The command."}
                },
                "required": ["command"],
            },
        },
    }
)

EPISODES: dict[str, cheatbox.Episode] = {}
"""Live episodes of this worker process, keyed by an id the loop hands to
the tool through verl's per-trajectory `tools_kwargs`."""


class Bash(BaseTool):
    """The one tool the model has. verl calls `execute` per tool call."""

    def __init__(self, config: dict, tool_schema=None):
        """verl builds one instance per worker from the tools YAML."""
        super().__init__(config, tool_schema or BASH_SCHEMA)
        parallel = int(os.environ.get("CHEATBOX_PARALLEL", "8"))
        self.slots = asyncio.Semaphore(parallel)

    async def execute(self, instance_id, parameters, agent_data=None, **_):
        """Run the command in this trajectory's Episode; return its output."""
        episode = EPISODES.get(agent_data.tools_kwargs.get("episode"))
        if episode is None:
            return (
                ToolResponse(text="Error: this episode has no sandbox."),
                0,
                {},
            )
        command = str(parameters.get("command", ""))
        async with self.slots:
            output = await asyncio.to_thread(episode.step, command)
        return ToolResponse(text=output), 0.0, {}


# --------------------------------------------------------------------- loop


@register("cheatbox")
class CheatboxLoop(ToolAgentLoop):
    """verl's tool loop, plus episode ownership and an output budget.

    Two hooks, nothing else: `run` (once per trajectory) is wrapped to
    open and grade an Episode, and `_handle_generating_state` (once per
    model call) is wrapped to cap what the model may still generate at
    the task's budget: the total number of tokens the model may generate
    across all its turns, the number its prompt states. The tokens, masks
    and tool dispatch stay entirely with verl.
    """

    def __init__(self, *args, **kwargs):
        """The `<think>` tag ids are needed to count reasoning tokens."""
        super().__init__(*args, **kwargs)
        self.think_ids = tuple(
            self.tokenizer.convert_tokens_to_ids(["<think>", "</think>"])
        )

    async def run(self, sampling_params, **kwargs):
        """Start the Episode, run verl's tool loop, grade, tear down.

        `kwargs` is the parquet row. We rebuild the Task from `extra_info`,
        and route the Episode to the tool through `tools_kwargs`. verl's
        `run` returns the finished `AgentLoopOutput`; we add the grade to
        it as `reward_score` and its metrics as `reward_extra_info`, and
        append the episode's numbers to the episode log.
        """
        extra_info = kwargs["extra_info"]
        task = TaskRecord(**strip(extra_info)).to_task()
        key = uuid.uuid4().hex
        episode = await asyncio.to_thread(cheatbox.Episode, task)
        EPISODES[key] = episode
        try:
            kwargs = {**kwargs, "tools_kwargs": {"episode": key}}
            output = await super().run(sampling_params, **kwargs)
            grade = await asyncio.to_thread(episode.grade)
        finally:
            EPISODES.pop(key, None)
            await asyncio.to_thread(episode.close)
        output.reward_score = grade.score
        output.extra_fields["reward_extra_info"] = grade.metrics
        EpisodeRecord(
            split=extra_info.get("split", "train"),
            reward=grade.score,
            total_output_tokens=sum(output.response_mask),
            total_reasoning_tokens=count_reasoning_tokens(
                output.prompt_ids + output.response_ids,
                [0] * len(output.prompt_ids) + output.response_mask,
                self.think_ids,
            ),
            total_input_tokens=len(output.prompt_ids)
            + len(output.response_mask)
            - sum(output.response_mask),
            nb_calls=episode.calls,
            black_read=episode.done,
            transcript=self.tokenizer.decode(
                output.prompt_ids + output.response_ids
            ),
        ).append()
        return output

    async def _handle_generating_state(self, agent_data, params, **kw):
        """Called by verl before each model turn. Stop once the budget is
        spent or the black file has been opened; otherwise let this turn
        generate at most what remains.

        `response_mask` marks model tokens with 1 and tool-output tokens with
        0, so its sum is exactly what the model has generated so far.
        """
        episode = EPISODES[agent_data.tools_kwargs["episode"]]
        used = sum(agent_data.response_mask)
        if episode.done or used >= episode.task.budget:
            return AgentState.TERMINATED
        params = {**params, "max_tokens": episode.task.budget - used}
        return await super()._handle_generating_state(agent_data, params, **kw)


def count_reasoning_tokens(
    ids: list[int], mask: list[int], think_ids: tuple[int, int]
) -> int:
    """Number of model-generated tokens (mask 1) that sit between a
    `<think>` and the next `</think>`. The opening tag may come from the
    prompt template, so both prompt and response tokens toggle the state.
    """
    start, end = think_ids
    inside = False
    count = 0
    for token, generated in zip(ids, mask):
        if token == start:
            inside = True
        elif token == end:
            inside = False
        elif inside and generated:
            count += 1
    return count


def strip(extra_info: dict) -> dict:
    """Drop the row-level keys so the rest matches TaskRecord's fields."""
    return {k: v for k, v in extra_info.items() if k not in ("index", "split")}


# -------------------------------------------------------------- episode log

EPISODE_LOG = os.environ.get("CHEATBOX_EPISODE_LOG")


@dataclass
class EpisodeRecord:
    """What one finished episode contributes to the env metrics."""

    split: str
    reward: float
    total_output_tokens: int
    total_reasoning_tokens: int
    total_input_tokens: int
    nb_calls: int
    black_read: bool
    transcript: str = ""

    def append(self) -> None:
        """Add this record to the episode log, if there is one."""
        if EPISODE_LOG:
            with open(EPISODE_LOG, "a") as log:
                log.write(json.dumps(asdict(self)) + "\n")


class EpisodeLog:
    """The reading side: hands out the records appended since the last
    `drain`, starting from the end of whatever the file held at start-up.
    """

    def __init__(self, path: str | None = None):
        """Remember the path (default: the episode log) and skip what is
        already there.
        """
        self.path = EPISODE_LOG if path is None else path
        self.offset = 0
        if self.path and os.path.exists(self.path):
            self.offset = os.path.getsize(self.path)

    def drain(self) -> list[EpisodeRecord]:
        """New records, oldest first; empty when there is no log."""
        if not self.path or not os.path.exists(self.path):
            return []
        with open(self.path) as log:
            log.seek(self.offset)
            lines = log.readlines()
            self.offset = log.tell()
        return [EpisodeRecord(**json.loads(x)) for x in lines if x.strip()]


# ----------------------------------------------------------------- progress
#
# verl prints one very long `step:N - key:value - key:value ...` line per
# training step (validation metrics ride along on the same line). We keep a
# handful of them, add the episode log's averages, print one line per group,
# and send the same numbers to wandb under `rollout_train/`, `rollout_valid/`,
# `training/` and `extra/`.


@dataclass
class RolloutMetrics:
    """Averages over one step's episodes of one split."""

    reward: float
    black_read: float
    total_output_tokens: float
    total_reasoning_tokens: float
    total_input_tokens: float
    nb_calls: float

    @classmethod
    def from_records(
        cls, records: list[EpisodeRecord]
    ) -> RolloutMetrics | None:
        """Average the records, or None when there are none."""
        if not records:
            return None
        n = len(records)
        return cls(
            sum(r.reward for r in records) / n,
            sum(r.black_read for r in records) / n,
            sum(r.total_output_tokens for r in records) / n,
            sum(r.total_reasoning_tokens for r in records) / n,
            sum(r.total_input_tokens for r in records) / n,
            sum(r.nb_calls for r in records) / n,
        )

    def line(self, step: int, group: str) -> str:
        """One log line."""
        return (
            f"step {step:>4}  {group}  reward {self.reward:.2f}"
            f"  black {self.black_read:.2f}"
            f"  out {self.total_output_tokens:.0f}"
            f"  think {self.total_reasoning_tokens:.0f}"
            f"  in {self.total_input_tokens:.0f}"
            f"  calls {self.nb_calls:.1f}"
        )


@dataclass
class TrainingMetrics:
    """What we keep from the policy update. `kl` exists only when the
    config has a KL term; `mismatch` is the KL between the sampler's and the
    trainer's token probabilities, which should stay near 0.001.
    """

    entropy: float
    pg_loss: float
    grad_norm: float
    kl: float | None = None
    mismatch: float | None = None

    KEYS = {
        "entropy": "actor/entropy",
        "pg_loss": "actor/pg_loss",
        "grad_norm": "actor/grad_norm",
    }
    OPTIONAL = {"kl": "actor/kl_loss", "mismatch": "rollout_corr/kl"}

    def line(self, step: int) -> str:
        """One log line."""
        kl = "" if self.kl is None else f"kl {self.kl:.4f}  "
        mismatch = (
            "" if self.mismatch is None else f"  mismatch {self.mismatch:.4f}"
        )
        return (
            f"step {step:>4}  training   {kl}"
            f"entropy {self.entropy:.3f}  pg_loss {self.pg_loss:.4f}"
            f"  grad_norm {self.grad_norm:.2f}{mismatch}"
        )


@dataclass
class ExtraMetrics:
    """Where the step's wall-clock time went."""

    generation_seconds: float
    training_seconds: float
    step_seconds: float
    validation_seconds: float | None = None

    KEYS = {
        "generation_seconds": "timing_s/gen",
        "training_seconds": "timing_s/update_actor",
        "step_seconds": "timing_s/step",
    }
    OPTIONAL = {"validation_seconds": "timing_s/testing"}

    def line(self, step: int) -> str:
        """One log line."""
        text = (
            f"step {step:>4}  extra      generation"
            f" {self.generation_seconds:.0f}s"
            f"  training {self.training_seconds:.0f}s"
            f"  step {self.step_seconds:.0f}s"
        )
        if self.validation_seconds is not None:
            text += f"  validation {self.validation_seconds:.0f}s"
        return text


@dataclass
class StepReport:
    """Everything we know about one step, by group."""

    step: int
    rollout_train: RolloutMetrics | None
    rollout_valid: RolloutMetrics | None
    training: TrainingMetrics | None
    extra: ExtraMetrics | None

    GROUPS = ("rollout_train", "rollout_valid", "training", "extra")

    def lines(self) -> list[str]:
        """One log line per group that has data."""
        lines = []
        if self.rollout_train:
            lines.append(self.rollout_train.line(self.step, "rollout_train"))
        if self.rollout_valid:
            lines.append(self.rollout_valid.line(self.step, "rollout_valid"))
        if self.training:
            lines.append(self.training.line(self.step))
        if self.extra:
            lines.append(self.extra.line(self.step))
        return lines

    def wandb_payload(self) -> dict[str, float]:
        """Flat `<group>/<metric>` numbers."""
        payload = {}
        for group in self.GROUPS:
            metrics = getattr(self, group)
            if metrics is None:
                continue
            for key, value in asdict(metrics).items():
                if value is not None:
                    payload[f"{group}/{key}"] = value
        return payload


STEP_LINE = re.compile(r"step:(\d+) - (.*)")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
RAY_PREFIX = re.compile(r"^(\(\w+ pid=\d+\)\s*)+")
ERROR_MARKS = (
    "Traceback (most recent call last)",
    "Error executing job",
    "srun: error",
    "DUE TO PREEMPTION",
    "Out Of Memory",
    "oom_kill",
)
RESUME_MARKS = ("Training from scratch", "Resuming from")


def parse_step(
    line: str, records: list[EpisodeRecord] = ()
) -> StepReport | None:
    """Turn one of verl's step lines plus the step's episode records into a
    StepReport, or None if the line is not a step line.
    """
    match = STEP_LINE.search(line)
    if not match:
        return None
    values = {}
    for item in match.group(2).split(" - "):
        key, _, value = item.partition(":")
        try:
            values[key] = float(re.sub(r"np\.\w+\(|\)", "", value))
        except ValueError:
            continue
    return StepReport(
        int(match.group(1)),
        RolloutMetrics.from_records([r for r in records if r.split == "train"]),
        RolloutMetrics.from_records([r for r in records if r.split == "val"]),
        pick(TrainingMetrics, values),
        pick(ExtraMetrics, values),
    )


def pick(cls, values: dict[str, float]):
    """Build `cls` from verl's metric names, or None if a required one is
    absent; `OPTIONAL` names are filled in when present.
    """
    if any(key not in values for key in cls.KEYS.values()):
        return None
    fields = {field: values[key] for field, key in cls.KEYS.items()}
    for field, key in getattr(cls, "OPTIONAL", {}).items():
        if key in values:
            fields[field] = values[key]
    return cls(**fields)


def clean(raw: str) -> str:
    """Strip colour codes and Ray's `(Actor pid=N)` prefixes."""
    return RAY_PREFIX.sub("", ANSI.sub("", raw.rstrip()))


def open_wandb():
    """The wandb run for this experiment, or None outside a training job.

    The run id includes the Slurm job id, so a requeued job resumes it;
    `WANDB_RUN_ID` overrides it, for a fresh run inside a held allocation.
    """
    name = os.environ.get("EXP")
    if not name or os.environ.get("WANDB_MODE") == "disabled":
        return None
    import wandb

    job = os.environ.get("SLURM_JOB_ID", "local")
    run_id = os.environ.get("WANDB_RUN_ID") or f"{name}-{job}"
    run = wandb.init(
        project="cheatbox",
        name=name,
        id=re.sub(r"[^A-Za-z0-9_-]", "-", run_id),
        resume="allow",
        config={k: os.environ.get(k) for k in ("MODEL", "DATA")},
    )
    print(f"wandb {run.url}", flush=True)
    return run


SAMPLE_COLUMNS = ("step", "kind", "reward", "calls", "transcript")


def samples(records: list[EpisodeRecord]) -> list[tuple[str, EpisodeRecord]]:
    """The episodes worth reading, each with why it was picked: a random
    rewarded one, a random unrewarded one, and one random episode whatever
    its reward.
    """
    if not records:
        return []
    groups = {
        "rewarded": [r for r in records if r.reward > 0],
        "unrewarded": [r for r in records if r.reward == 0],
        "random": records,
    }
    return [(kind, random.choice(g)) for kind, g in groups.items() if g]


def sample_tables(step: int, records: list[EpisodeRecord]) -> dict:
    """One wandb table of transcripts per split that has records."""
    import wandb

    tables = {}
    for split, group in (("train", "rollout_train"), ("val", "rollout_valid")):
        rows = [
            [step, kind, r.reward, r.nb_calls, r.transcript]
            for kind, r in samples([e for e in records if e.split == split])
        ]
        if rows:
            tables[f"samples/{group}"] = wandb.Table(
                columns=list(SAMPLE_COLUMNS), data=rows
            )
    return tables


def progress(stream: Iterable[str], out: TextIO = sys.stdout) -> None:
    """Filter verl's console output into step lines and errors."""
    run = open_wandb()
    episodes = EpisodeLog()
    for raw in stream:
        line = clean(raw)
        if not STEP_LINE.search(line):
            if "WARNING" in line or "UserWarning" in line:
                continue
            if any(mark in line for mark in ERROR_MARKS + RESUME_MARKS):
                print(line[:300], file=out, flush=True)
            continue
        records = episodes.drain()
        if report := parse_step(line, records):
            for text in report.lines():
                print(text, file=out, flush=True)
            if run and (payload := report.wandb_payload()):
                payload.update(sample_tables(report.step, records))
                run.log(payload, step=report.step)
    if run:
        run.finish()


# ---------------------------------------------------------------------- cli

SOKOBAN = [
    "min_w=6",
    "max_w=6",
    "min_h=6",
    "max_h=6",
    "min_boxes=2",
    "max_boxes=2",
    "max_depth=5",
]


def parse_option(item: str) -> tuple[str, object]:
    """`key=value` with the value read as JSON when it parses, else text."""
    key, _, value = item.partition("=")
    try:
        return key, json.loads(value)
    except ValueError:
        return key, value


def main() -> None:
    """`data` writes the parquet files; `progress` filters stdin."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    data = commands.add_parser("data", help="write train/val parquet files")
    data.add_argument("--out", default="data/sokoban")
    data.add_argument(
        "--source",
        default="rg:sokoban",
        help="rg:<reasoning-gym dataset> or mbpp",
    )
    data.add_argument(
        "--config",
        nargs="*",
        metavar="KEY=VALUE",
        help="source options (Sokoban default: 6x6, 2 boxes, depth 5)",
    )
    data.add_argument("-n", type=int, default=2000, help="problems")
    data.add_argument("--seed", type=int, default=0)
    data.add_argument(
        "--no-cheat",
        action="store_true",
        help="train cheat files hold the val decoy: nothing to copy",
    )
    data.add_argument(
        "--budget",
        type=int,
        default=cheatbox.prompts.BUDGET_TOKENS,
        help="output tokens per episode, stated in the prompt and enforced",
    )
    commands.add_parser("progress", help="filter verl's output from stdin")
    args = parser.parse_args()
    if args.command == "data":
        items = args.config
        if items is None:
            items = SOKOBAN if args.source == "rg:sokoban" else []
        config = dict(parse_option(item) for item in items)
        write_dataset(
            Path(args.out),
            args.source,
            config,
            args.n,
            args.budget,
            args.seed,
            not args.no_cheat,
        )
    else:
        progress(sys.stdin)


if __name__ == "__main__":
    main()
