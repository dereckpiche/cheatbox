import verl.experimental.agent_loop.agent_loop as agent_loop

import cheatbox
import training_example as te


def test_task_record_round_trips_through_a_row():
    task = cheatbox.Task(
        7,
        "sys",
        "usr",
        {
            "solution.txt": "",
            "file_a.txt": "task\nq\n",
            "file_b.txt": "Black\nabc\n",
        },
        "rg:sokoban",
        {"min_w": 6},
        {"metadata": {"gamestr": "x"}},
        "U",
        "file_b.txt",
        budget=4096,
    )
    row = te.TaskRecord.from_task(task).to_row(index=3, split="train")
    assert row["prompt"][0]["content"] == "sys"
    assert row["extra_info"]["split"] == "train"
    assert row["extra_info"]["budget"] == 4096
    assert te.TaskRecord(**te.strip(row["extra_info"])).to_task() == task


def test_loop_is_registered():
    assert "cheatbox" in agent_loop._agent_loop_registry
