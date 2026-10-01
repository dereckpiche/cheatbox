"""A container with a host-visible workspace seeded with the task's files."""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

from cheatbox.container import Exec, container

IMAGE = os.environ.get("CHEATBOX_IMAGE", "docker.io/library/python:3.12-slim")
ROOT = Path(os.environ.get("CHEATBOX_ROOT", "/tmp/cheatbox"))


class Sandbox:
    """One episode's container: bash in `/work`, which starts out holding
    `files`.
    """

    def __init__(
        self,
        files: dict[str, str],
        *,
        image: str = IMAGE,
        root: Path = ROOT,
        timeout: int = 30,
    ):
        """Create the workspace under `root`, write the files, then start
        the container.
        """
        self.timeout = timeout
        self.dir = root / f"episode-{uuid.uuid4().hex}"
        self.work = self.dir / "work"
        self.work.mkdir(parents=True)
        for name, text in files.items():
            self.write(name, text)
        try:
            self.container = container(
                image,
                cwd="/work",
                mounts=[f"{self.work}:/work"],
                env={"PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1"},
            )
        except Exception:
            shutil.rmtree(self.dir, ignore_errors=True)
            raise

    def run(self, command: str, timeout: int | None = None) -> Exec:
        """Run a bash command in the workspace."""
        return self.container.exec(command, timeout or self.timeout)

    def read(self, path: str) -> str | None:
        """Read a workspace file from the host; None if it does not exist or
        escapes the workspace.
        """
        try:
            return self._path(path).read_text(errors="replace")
        except (OSError, ValueError):
            return None

    def write(self, path: str, text: str) -> None:
        """Write a workspace file from the host, creating parent directories."""
        target = self._path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)

    def close(self) -> None:
        """Remove the container and delete the workspace."""
        self.container.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def _path(self, path: str) -> Path:
        """Resolve a workspace-relative path, refusing anything that escapes
        the workspace.
        """
        target = (self.work / path).resolve()
        if not target.is_relative_to(self.work.resolve()):
            raise ValueError(f"{path} is outside the workspace")
        return target
