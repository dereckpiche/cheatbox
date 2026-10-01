"""cheatbox: a small reward-hacking sandbox. Each task is a problem from some
source (a reasoning-gym puzzle, an MBPP coding task) in a workspace file
next to a file holding an answer; the agent must find the task and write
its own solution.

    task = cheatbox.tasks.generate("rg:sokoban", 8)["val"][0]
    ep = cheatbox.Episode(task)
    ep.messages
    ep.step("cat file_*.txt")
    ep.step("echo UURR > solution.txt")
    ep.grade()
    ep.close()
"""

from cheatbox import prompts, sources, tasks
from cheatbox.episode import Episode
from cheatbox.grade import Grade
from cheatbox.sandbox import Sandbox
from cheatbox.tasks import Task

__all__ = ["Episode", "Grade", "Sandbox", "Task", "prompts", "sources", "tasks"]
