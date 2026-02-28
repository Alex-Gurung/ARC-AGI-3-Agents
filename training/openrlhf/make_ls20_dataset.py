"""Build a tiny prompt dataset for OpenRLHF LS20 training.

Usage:
    uv run python training/openrlhf/make_ls20_dataset.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_PROMPT = """You are controlling an ARC-AGI game agent.
Game: ls20

Goal:
- Learn the game mechanics and complete as many levels as possible.
- Prefer actions that reveal new information when uncertain.
- If the path forward is unclear, test an assumption (action effect, object role, or goal condition).

Action format requirements:
- Think briefly, then end with exactly one final line:
  ANSWER: <ACTION>
- Valid actions include RESET, ACTION1..ACTION6 (when available).
- If using ACTION6, use coordinates:
  ANSWER: ACTION6 x y
""".strip()


def build_rows(num_rows: int, prompt: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for i in range(num_rows):
        rows.append(
            {
                "observation": prompt,
                "label": "",
                "game_id": "ls20",
                "seed_id": str(i),
            }
        )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create LS20 prompt dataset for OpenRLHF")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training/openrlhf/data/ls20_prompts.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--num-rows",
        type=int,
        default=512,
        help="Number of prompts to generate",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Optional text file with a custom prompt template",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_rows <= 0:
        raise ValueError("--num-rows must be > 0")

    prompt = DEFAULT_PROMPT
    if args.prompt_file:
        prompt = args.prompt_file.read_text(encoding="utf-8").strip()

    rows = build_rows(args.num_rows, prompt)
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
