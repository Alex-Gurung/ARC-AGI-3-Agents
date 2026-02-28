"""Online grouped-rollout training loop scaffold for ls20.

This orchestrates on-policy-style iterations:
1) collect grouped rollout decisions with current model
2) run a configurable GRPO update command (veRL adapter)
3) use updated model for next iteration
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents.templates.loop_agent.memory import Memory
from training.verl.agent_loop_ls20 import LS20GroupedRunner

logger = logging.getLogger(__name__)


@dataclass
class OnlineConfig:
    game_id: str = "ls20"
    model: str = os.environ.get("TRAIN_MODEL", "google/gemma-3-4b-it")
    iterations: int = int(os.environ.get("ONLINE_ITERS", "3"))
    attempts_per_iteration: int = int(os.environ.get("ATTEMPTS_PER_ITER", "8"))
    max_steps: int = int(os.environ.get("MAX_STEPS", "50"))
    seed: int = int(os.environ.get("SEED", "0"))
    output_dir: Path = Path(os.environ.get("OUTPUT_DIR", "training/verl/data/online"))
    update_cmd_template: str = os.environ.get("VERL_GRPO_UPDATE_CMD", "").strip()
    carry_memory_across_attempts: bool = True
    require_update_step: bool = os.environ.get("REQUIRE_UPDATE_STEP", "true").lower() == "true"


class VerlUpdateAdapter:
    """Runs external update command for one iteration if configured."""

    def __init__(self, template: str) -> None:
        self.template = template

    def update(
        self,
        *,
        model: str,
        rollout_jsonl: Path,
        output_dir: Path,
        iteration: int,
        require_update: bool = False,
    ) -> str:
        """Run update and return next model/checkpoint path.

        If no command template is provided, acts as no-op and returns model.
        """
        if not self.template:
            if require_update:
                raise RuntimeError(
                    "REQUIRE_UPDATE_STEP=true but VERL_GRPO_UPDATE_CMD is not set."
                )
            logger.warning(
                "VERL_GRPO_UPDATE_CMD is not set; skipping optimizer step (collector-only mode)."
            )
            return model

        iteration_dir = output_dir / f"iter_{iteration:04d}"
        iteration_dir.mkdir(parents=True, exist_ok=True)
        cmd = self.template.format(
            model=model,
            rollout_jsonl=str(rollout_jsonl),
            rollouts=str(rollout_jsonl),
            output_dir=str(output_dir),
            iteration_dir=str(iteration_dir),
            iteration=iteration,
        )
        logger.info("Running update command: %s", cmd)
        subprocess.run(shlex.split(cmd), check=True)

        # Convention: updater writes latest checkpoint path to one of these files.
        model_ptrs = [
            iteration_dir / "latest_model.txt",
            output_dir / "latest_model.txt",
        ]
        for model_ptr in model_ptrs:
            if model_ptr.exists():
                next_model = model_ptr.read_text(encoding="utf-8").strip()
                if next_model:
                    return next_model
        # Fallback: stay on same model if pointer not emitted.
        logger.warning(
            "Update command completed but latest_model.txt missing; continuing with previous model."
        )
        return model


def run_online(config: OnlineConfig) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    updater = VerlUpdateAdapter(config.update_cmd_template)
    current_model = config.model
    memory_pool: Memory | None = None
    summary: dict[str, Any] = {
        "game_id": config.game_id,
        "iterations": [],
    }

    for iteration in range(config.iterations):
        iter_dir = config.output_dir / f"iter_{iteration:04d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        rollout_jsonl = iter_dir / "rollouts.jsonl"
        if rollout_jsonl.exists():
            rollout_jsonl.unlink()

        runner = LS20GroupedRunner(
            model=current_model,
            game_id=config.game_id,
            max_steps=config.max_steps,
            seed=config.seed + iteration,
        )
        iter_stats: dict[str, Any] = {
            "iteration": iteration,
            "model_in": current_model,
            "attempts": [],
        }

        for attempt_idx in range(config.attempts_per_iteration):
            attempt_id = f"iter{iteration:04d}_attempt{attempt_idx:04d}"
            episode, final_memory = runner.run_attempt_with_memory(
                attempt_id=attempt_id,
                output_jsonl=rollout_jsonl,
                base_memory=memory_pool if config.carry_memory_across_attempts else None,
            )
            iter_stats["attempts"].append(episode.to_dict())

            # Carry last canonical memory across attempts when enabled.
            if config.carry_memory_across_attempts:
                memory_pool = final_memory

        next_model = updater.update(
            model=current_model,
            rollout_jsonl=rollout_jsonl,
            output_dir=config.output_dir,
            iteration=iteration,
            require_update=config.require_update_step,
        )
        iter_stats["model_out"] = next_model
        summary["iterations"].append(iter_stats)
        current_model = next_model

        summary_path = config.output_dir / "online_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game-id", type=str, default=os.environ.get("GAME_ID", "ls20"))
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("TRAIN_MODEL", "google/gemma-3-4b-it"),
    )
    parser.add_argument("--iterations", type=int, default=int(os.environ.get("ONLINE_ITERS", "3")))
    parser.add_argument(
        "--attempts-per-iter",
        type=int,
        default=int(os.environ.get("ATTEMPTS_PER_ITER", "8")),
    )
    parser.add_argument("--max-steps", type=int, default=int(os.environ.get("MAX_STEPS", "50")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "0")))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("OUTPUT_DIR", "training/verl/data/online")),
    )
    parser.add_argument(
        "--no-memory-carry",
        action="store_true",
        help="Disable memory carryover across attempts in same iteration",
    )
    parser.add_argument(
        "--allow-collector-only",
        action="store_true",
        help="Allow rollout collection without running update step",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    config = OnlineConfig(
        game_id=args.game_id,
        model=args.model,
        iterations=args.iterations,
        attempts_per_iteration=args.attempts_per_iter,
        max_steps=args.max_steps,
        seed=args.seed,
        output_dir=args.output_dir,
        carry_memory_across_attempts=not args.no_memory_carry,
        require_update_step=not args.allow_collector_only,
    )
    summary = run_online(config)
    print(json.dumps({"online_summary_path": str(config.output_dir / "online_summary.json"), "iterations": len(summary["iterations"])}, ensure_ascii=True))


if __name__ == "__main__":
    main()
