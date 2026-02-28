from pathlib import Path

from training.openrlhf.online_grpo_ls20 import resolve_next_model


def test_resolve_next_model_prefers_pointer_file(tmp_path: Path) -> None:
    iteration_dir = tmp_path / "iter_0001"
    iteration_dir.mkdir(parents=True, exist_ok=True)
    save_path = iteration_dir / "model"
    ckpt_path = iteration_dir / "ckpt"
    ckpt_path.mkdir(parents=True, exist_ok=True)
    (iteration_dir / "latest_model.txt").write_text("checkpoint/model_A", encoding="utf-8")

    model = resolve_next_model(
        iteration_dir=iteration_dir,
        save_path=save_path,
        ckpt_path=ckpt_path,
        fallback_model="google/gemma-3-4b-it",
    )
    assert model == "checkpoint/model_A"


def test_resolve_next_model_uses_save_dir_when_model_markers_exist(tmp_path: Path) -> None:
    iteration_dir = tmp_path / "iter_0002"
    save_path = iteration_dir / "model"
    ckpt_path = iteration_dir / "ckpt"
    save_path.mkdir(parents=True, exist_ok=True)
    ckpt_path.mkdir(parents=True, exist_ok=True)
    (save_path / "config.json").write_text("{}", encoding="utf-8")

    model = resolve_next_model(
        iteration_dir=iteration_dir,
        save_path=save_path,
        ckpt_path=ckpt_path,
        fallback_model="fallback/model",
    )
    assert model == str(save_path)


def test_resolve_next_model_falls_back_when_no_outputs(tmp_path: Path) -> None:
    iteration_dir = tmp_path / "iter_0003"
    save_path = iteration_dir / "model"
    ckpt_path = iteration_dir / "ckpt"
    iteration_dir.mkdir(parents=True, exist_ok=True)

    model = resolve_next_model(
        iteration_dir=iteration_dir,
        save_path=save_path,
        ckpt_path=ckpt_path,
        fallback_model="fallback/model",
    )
    assert model == "fallback/model"
