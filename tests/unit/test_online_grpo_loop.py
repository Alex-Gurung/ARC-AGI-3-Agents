from pathlib import Path

from training.verl.online_grpo_ls20 import VerlUpdateAdapter


def test_update_adapter_noop_when_template_missing(tmp_path: Path) -> None:
    adapter = VerlUpdateAdapter("")
    model = adapter.update(
        model="google/gemma-3-4b-it",
        rollout_jsonl=tmp_path / "rollouts.jsonl",
        output_dir=tmp_path,
        iteration=0,
    )
    assert model == "google/gemma-3-4b-it"


def test_update_adapter_requires_command_when_flag_enabled(tmp_path: Path) -> None:
    adapter = VerlUpdateAdapter("")
    try:
        adapter.update(
            model="google/gemma-3-4b-it",
            rollout_jsonl=tmp_path / "rollouts.jsonl",
            output_dir=tmp_path,
            iteration=0,
            require_update=True,
        )
    except RuntimeError as err:
        assert "VERL_GRPO_UPDATE_CMD" in str(err)
    else:  # pragma: no cover - should not happen
        raise AssertionError("expected RuntimeError when require_update=True and template is empty")


def test_update_adapter_reads_latest_model_pointer(tmp_path: Path) -> None:
    template = (
        "python -c \"from pathlib import Path; "
        "d=Path('{output_dir}')/f'iter_{iteration:04d}'; "
        "d.mkdir(parents=True, exist_ok=True); "
        "(d/'latest_model.txt').write_text('checkpoint/new_model')\""
    )
    adapter = VerlUpdateAdapter(template)
    model = adapter.update(
        model="google/gemma-3-4b-it",
        rollout_jsonl=tmp_path / "rollouts.jsonl",
        output_dir=tmp_path,
        iteration=1,
    )
    assert model == "checkpoint/new_model"
