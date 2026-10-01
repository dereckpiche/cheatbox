import training_example as te

STEP = (
    "\x1b[36m(TaskRunnerV1 pid=1)\x1b[0m step:3 - critic/score/mean:0.25"
    " - actor/kl_loss:0.001 - actor/entropy:0.412 - actor/pg_loss:-0.1"
    " - actor/grad_norm:0.35 - timing_s/gen:20.4 - timing_s/update_actor:21.2"
    " - timing_s/step:62.0"
)
VAL_STEP = STEP.replace("step:3", "step:10") + " - timing_s/testing:16.1"


def record(split, reward, calls=5, black=False):
    return te.EpisodeRecord(split, reward, 700, 500, 900, calls, black)


def test_step_with_train_episodes_gives_three_lines():
    records = [record("train", 1.0), record("train", 0.0, calls=3, black=True)]
    report = te.parse_step(te.clean(STEP), records)
    assert report.rollout_valid is None
    assert report.lines() == [
        "step    3  rollout_train  reward 0.50  black 0.50  out 700  think 500"
        "  in 900  calls 4.0",
        "step    3  training   kl 0.0010  entropy 0.412  pg_loss -0.1000"
        "  grad_norm 0.35",
        "step    3  extra      generation 20s  training 21s  step 62s",
    ]
    payload = report.wandb_payload()
    assert payload["rollout_train/reward"] == 0.5
    assert payload["rollout_train/black_read"] == 0.5
    assert payload["extra/step_seconds"] == 62.0
    assert "extra/validation_seconds" not in payload


def test_validation_step_gives_all_four_groups():
    records = [record("train", 1.0), record("val", 0.0)]
    report = te.parse_step(VAL_STEP, records)
    lines = report.lines()
    assert len(lines) == 4
    assert lines[1].startswith(
        "step   10  rollout_valid  reward 0.00  black 0.00"
    )
    assert lines[3].endswith("step 62s  validation 16s")
    assert report.wandb_payload()["rollout_valid/reward"] == 0.0


def test_step_line_without_records_still_reports_training():
    report = te.parse_step(STEP)
    assert report.rollout_train is None and report.training is not None


def test_episode_log_drains_only_new_lines(tmp_path, monkeypatch):
    path = tmp_path / "episodes.jsonl"
    monkeypatch.setattr(te, "EPISODE_LOG", str(path))
    record("val", 1.0).append()
    log = te.EpisodeLog(str(path))
    assert log.drain() == []
    record("train", 1.0).append()
    record("train", 0.0).append()
    assert [r.split for r in log.drain()] == ["train", "train"]
    assert log.drain() == []


def test_samples_take_a_rewarded_an_unrewarded_and_a_random_one():
    records = [record("train", 0.0), record("train", 0.0)]
    records += [record("train", 1.0), record("train", 0.5)]
    chosen = te.samples(records)
    assert [kind for kind, _ in chosen] == ["rewarded", "unrewarded", "random"]
    assert chosen[0][1].reward > 0 and chosen[1][1].reward == 0
    assert chosen[2][1] in records
    only = [record("train", 0.0)]
    assert [kind for kind, _ in te.samples(only)] == ["unrewarded", "random"]
    assert te.samples([]) == []


def test_ray_prefix_and_warnings():
    line = "(vLLMHttpServer pid=1) (Worker pid=2) WARNING [x] Traceback (most recent call last):"
    assert te.clean(line).startswith("WARNING")
    assert te.parse_step("nothing here") is None


def test_progress_drains_only_at_step_lines(tmp_path, monkeypatch, capsys):
    path = tmp_path / "episodes.jsonl"
    monkeypatch.setattr(te, "EPISODE_LOG", str(path))
    monkeypatch.delenv("EXP", raising=False)
    import io

    def stream():
        yield "(TaskRunnerV1 pid=1) some chatter\n"
        record("train", 1.0).append()
        yield "(TaskRunnerV1 pid=1) more chatter\n"
        yield STEP + "\n"

    out = io.StringIO()
    te.progress(stream(), out)
    assert out.getvalue().startswith("step    3  rollout_train  reward 1.00")


def test_count_reasoning_tokens_handles_prompt_side_open_tag():
    think, end = 7, 8
    ids = [1, think, 2, 3, end, 4, 5, think, 6, end]
    mask = [0, 0, 1, 1, 1, 1, 0, 1, 1, 1]
    assert te.count_reasoning_tokens(ids, mask, (think, end)) == 3


def test_sampler_trainer_mismatch_is_reported():
    step = STEP + " - rollout_corr/kl:0.0012"
    report = te.parse_step(te.clean(step), [record("train", 1.0)])
    assert report.lines()[1].endswith("grad_norm 0.35  mismatch 0.0012")
    assert report.wandb_payload()["training/mismatch"] == 0.0012
