"""One container per episode: podman through mini-swe-agent's
DockerEnvironment, or Apptainer on clusters without podman."""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")

from minisweagent.environments.docker import DockerEnvironment  # noqa: E402
from minisweagent.exceptions import Submitted  # noqa: E402

logging.getLogger("minisweagent.environment").setLevel(logging.WARNING)
LOG = logging.getLogger(__name__)

QUIET_RUNTIME = str(Path(__file__).with_name("bin") / "podman-quiet")
RUNTIME = os.environ.get("CHEATBOX_RUNTIME", "podman")
START_ATTEMPTS = 5
START_PARALLEL = int(os.environ.get("CHEATBOX_START_PARALLEL", "8"))
STARTS = threading.BoundedSemaphore(START_PARALLEL)


@dataclass
class Exec:
    """Result of one command: interleaved stdout and stderr, exit code, and
    whether it was killed on timeout.
    """

    output: str
    returncode: int
    timed_out: bool


class Container:
    """A long-lived container (`run -d ... sleep`) that executes each command
    with `exec`. Files persist between commands; shell state does not.
    """

    def __init__(
        self,
        image: str,
        *,
        cwd: str,
        mounts: list[str],
        env: dict[str, str],
        memory: str = "1g",
        pids: int = 256,
        lifetime: str = "2h",
    ):
        """Start the container with no network, memory and pid limits, the
        given bind mounts and environment variables. Rootless podman fails
        to start containers when many start at once, so at most
        `START_PARALLEL` start concurrently per process, with retries.
        """
        run_args = ["--rm", "--network", "none", "--memory", memory]
        run_args += ["--pids-limit", str(pids)]
        for mount in mounts:
            run_args += ["-v", mount]
        for attempt in range(1, START_ATTEMPTS + 1):
            try:
                with STARTS:
                    self.env = DockerEnvironment(
                        image=image,
                        cwd=cwd,
                        env=env,
                        executable=QUIET_RUNTIME,
                        run_args=run_args,
                        container_timeout=lifetime,
                        interpreter=["bash", "-c"],
                    )
                return
            except subprocess.CalledProcessError as e:
                LOG.warning(
                    "podman run failed (attempt %d): %s",
                    attempt,
                    (e.stderr or "").strip()[-300:],
                )
                if attempt == START_ATTEMPTS:
                    raise
                time.sleep(attempt)

    @property
    def name(self) -> str:
        """The container id, for inspection and tests."""
        return self.env.container_id

    def exec(self, command: str, timeout: int = 30) -> Exec:
        """Run `command` with bash inside the container, killing it after
        `timeout` seconds.
        """
        guarded = f"timeout -s KILL {timeout} bash -c {shlex.quote(command)}"
        try:
            r = self.env.execute({"command": guarded}, timeout=timeout + 15)
        except Submitted:
            return Exec("", 0, False)
        timed_out = r["returncode"] in (137, -1)
        output = r["output"] or ""
        if timed_out:
            output += f"\n[timed out after {timeout}s]"
        return Exec(output, r["returncode"], timed_out)

    def close(self) -> None:
        """Stop and remove the container, under the same limit as starts."""
        with STARTS:
            self.env.cleanup()


class ApptainerContainer:
    """The same interface on Apptainer. There is no long-lived container:
    each command is one `apptainer exec` with the same binds, so files
    persist through the bind-mounted workspace and nothing else does.
    Memory and pid limits need cgroup delegation, which rootless
    Apptainer lacks, so only the kill timeout bounds a command.
    """

    def __init__(
        self,
        image: str,
        *,
        cwd: str,
        mounts: list[str],
        env: dict[str, str],
        memory: str = "1g",
        pids: int = 256,
        lifetime: str = "2h",
    ):
        """Remember the exec arguments; nothing starts until the first
        command.
        """
        self.image = image
        self.name = f"apptainer-{os.getpid()}-{id(self):x}"
        self.args = ["--containall", "--no-home", "--net", "--network", "none"]
        self.args += ["--pwd", cwd]
        for mount in mounts:
            self.args += ["--bind", mount]
        for key, value in env.items():
            self.args += ["--env", f"{key}={value}"]

    def exec(self, command: str, timeout: int = 30) -> Exec:
        """Run `command` with bash inside the image, killing it after
        `timeout` seconds.
        """
        guarded = f"timeout -s KILL {timeout} bash -c {shlex.quote(command)}"
        cmd = [RUNTIME, "--silent", "exec", *self.args, self.image]
        cmd += ["bash", "-c", guarded]
        try:
            r = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                timeout=timeout + 15,
            )
        except subprocess.TimeoutExpired as e:
            output = e.stdout.decode(errors="replace") if e.stdout else ""
            return Exec(output + f"\n[timed out after {timeout}s]", -1, True)
        timed_out = r.returncode == 137
        output = r.stdout or ""
        if timed_out:
            output += f"\n[timed out after {timeout}s]"
        return Exec(output, r.returncode, timed_out)

    def close(self) -> None:
        """Nothing to stop: no container outlives its command."""


container = ApptainerContainer if RUNTIME == "apptainer" else Container
"""The backend the sandbox uses, chosen by `CHEATBOX_RUNTIME`."""
