"""Create minimal prompts dataset for veRL ls20 grouped rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_rows(num_rows: int, game_id: str, max_steps: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx in range(num_rows):
        rows.append(
            {
                "id": f"{game_id}_{idx:06d}",
                "game_id": game_id,
                "max_steps": max_steps,
                "observation": (
                    "You are training a LoopAgent-style ARC policy. "
                    "Use Curiosity/Learner/Solver behavior conditioned by mode. "
                    "Return formatted ANSWER outputs only."
                ),
                "label": "",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training/verl/data/ls20_prompts.jsonl"),
    )
    parser.add_argument("--num-rows", type=int, default=512)
    parser.add_argument("--game-id", type=str, default="ls20")
    parser.add_argument("--max-steps", type=int, default=200)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = build_rows(args.num_rows, args.game_id, args.max_steps)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(f"Wrote {len(rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()
