"""Iterative on-policy OpenRLHF GRPO runner for ls20."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class OpenRLHFOnlineConfig:
    """Configuration for iterative OpenRLHF training."""

    model: str = os.environ.get("MODEL", "google/gemma-3-1b-it")
    iterations: int = int(os.environ.get("ONLINE_ITERS", "3"))
    max_samples_per_iter: int = int(os.environ.get("MAX_SAMPLES_PER_ITER", "1024"))
    dataset_path: Path = Path(
        os.environ.get("DATASET_PATH", "training/openrlhf/data/ls20_prompts.jsonl")
    )
    output_dir: Path = Path(
        os.environ.get("OUTPUT_DIR", "training/openrlhf/data/online")
    )
    runner_script: Path = Path(
        os.environ.get("OPENRLHF_RUN_SCRIPT", "training/openrlhf/run_grpo_ls20_1gpu.sh")
    )
    n_samples_per_prompt: int = int(os.environ.get("N_SAMPLES_PER_PROMPT", "4"))
    rollout_batch_size: int = int(os.environ.get("ROLLOUT_BATCH_SIZE", "8"))
    train_batch_size: int = int(os.environ.get("TRAIN_BATCH_SIZE", "32"))
    max_steps_per_episode: int = int(os.environ.get("MAX_STEPS_PER_EPISODE", "200"))
    vllm_gpu_util: float = float(os.environ.get("VLLM_GPU_UTIL", "0.5"))


def _ensure_dataset(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "uv",
        "run",
        "python",
        "training/openrlhf/make_ls20_dataset.py",
        "--output",
        str(path),
        "--num-rows",
        "512",
    ]
    logger.info("Dataset missing; generating with: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _looks_like_model_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    markers = ("config.json", "tokenizer.json", "model.safetensors", "pytorch_model.bin")
    return any((path / marker).exists() for marker in markers)


def _latest_dir(root: Path) -> Path | None:
    if not root.exists():
        return None
    dirs = [p for p in root.rglob("*") if p.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda p: p.stat().st_mtime)


def resolve_next_model(
    *,
    iteration_dir: Path,
    save_path: Path,
    ckpt_path: Path,
    fallback_model: str,
) -> str:
    """Resolve model path to use for next iteration."""
    pointer_candidates = [
        iteration_dir / "latest_model.txt",
        iteration_dir.parent / "latest_model.txt",
    ]
    for pointer in pointer_candidates:
        if pointer.exists():
            value = pointer.read_text(encoding="utf-8").strip()
            if value:
                return value

    if _looks_like_model_dir(save_path):
        return str(save_path)

    newest_ckpt = _latest_dir(ckpt_path)
    if newest_ckpt:
        return str(newest_ckpt)

    return fallback_model


def run_online(config: OpenRLHFOnlineConfig) -> dict[str, Any]:
    """Run iterative OpenRLHF GRPO jobs and return summary metadata."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_dataset(config.dataset_path)
    current_model = config.model
    summary: dict[str, Any] = {
        "backend": "openrlhf",
        "dataset_path": str(config.dataset_path),
        "iterations": [],
    }

    for iteration in range(config.iterations):
        iteration_dir = config.output_dir / f"iter_{iteration:04d}"
        save_path = iteration_dir / "model"
        ckpt_path = iteration_dir / "ckpt"
        iteration_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env.update(
            {
                "MODEL": current_model,
                "DATASET_PATH": str(config.dataset_path),
                "SAVE_PATH": str(save_path),
                "CKPT_PATH": str(ckpt_path),
                "MAX_SAMPLES": str(config.max_samples_per_iter),
                "N_SAMPLES_PER_PROMPT": str(config.n_samples_per_prompt),
                "ROLLOUT_BATCH_SIZE": str(config.rollout_batch_size),
                "TRAIN_BATCH_SIZE": str(config.train_batch_size),
                "MAX_STEPS_PER_EPISODE": str(config.max_steps_per_episode),
                "VLLM_GPU_UTIL": str(config.vllm_gpu_util),
            }
        )

        cmd = ["bash", str(config.runner_script)]
        logger.info("Starting OpenRLHF iteration %d with model=%s", iteration, current_model)
        subprocess.run(cmd, check=True, env=env)

        next_model = resolve_next_model(
            iteration_dir=iteration_dir,
            save_path=save_path,
            ckpt_path=ckpt_path,
            fallback_model=current_model,
        )
        (iteration_dir / "latest_model.txt").write_text(
            next_model, encoding="utf-8"
        )
        (config.output_dir / "latest_model.txt").write_text(
            next_model, encoding="utf-8"
        )

        summary["iterations"].append(
            {
                "iteration": iteration,
                "model_in": current_model,
                "model_out": next_model,
                "save_path": str(save_path),
                "ckpt_path": str(ckpt_path),
            }
        )
        current_model = next_model

    summary_path = config.output_dir / "online_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("MODEL", "google/gemma-3-1b-it"),
    )
    parser.add_argument("--iterations", type=int, default=int(os.environ.get("ONLINE_ITERS", "3")))
    parser.add_argument(
        "--max-samples-per-iter",
        type=int,
        default=int(os.environ.get("MAX_SAMPLES_PER_ITER", "1024")),
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path(os.environ.get("DATASET_PATH", "training/openrlhf/data/ls20_prompts.jsonl")),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("OUTPUT_DIR", "training/openrlhf/data/online")),
    )
    parser.add_argument(
        "--runner-script",
        type=Path,
        default=Path(
            os.environ.get(
                "OPENRLHF_RUN_SCRIPT",
                "training/openrlhf/run_grpo_ls20_1gpu.sh",
            )
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    config = OpenRLHFOnlineConfig(
        model=args.model,
        iterations=args.iterations,
        max_samples_per_iter=args.max_samples_per_iter,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        runner_script=args.runner_script,
    )
    summary = run_online(config)
    print(
        json.dumps(
            {
                "iterations": len(summary["iterations"]),
                "summary_path": str(config.output_dir / "online_summary.json"),
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
