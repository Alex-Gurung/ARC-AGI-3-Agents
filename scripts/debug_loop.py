"""Interactive Rich TUI debugger for the LoopAgent.

Step through the agent's mode-routed loop on a live game,
watching the grid, memory, LLM prompts/outputs, and mode transitions
in real-time.

Usage:
    uv run python scripts/debug_loop.py --game <game_id>
    uv run python scripts/debug_loop.py --game <game_id> --delay 0.5   # auto-run
"""

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from dotenv import load_dotenv
from rich.color import Color
from rich.columns import Columns
from rich.console import Console, Group
from rich.panel import Panel
from rich.style import Style
from rich.table import Table
from rich.text import Text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv(dotenv_path=".env.example")
load_dotenv(dotenv_path=".env", override=True)

# Suppress noisy loggers — we show everything in the TUI instead.
logging.basicConfig(level=logging.WARNING)
logging.getLogger("agents").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

console = Console()

# ---------------------------------------------------------------------------
# ARC color palette  (0-15 → RGB)
# ---------------------------------------------------------------------------
ARC_PALETTE: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 0),          # black
    1: (0, 116, 217),      # blue
    2: (255, 65, 54),      # red
    3: (46, 204, 64),      # green
    4: (255, 220, 0),      # yellow
    5: (170, 170, 170),    # gray
    6: (240, 18, 190),     # magenta/pink
    7: (255, 133, 27),     # orange
    8: (127, 219, 255),    # light blue
    9: (135, 12, 37),      # maroon
    10: (255, 255, 255),   # white
    11: (104, 195, 163),   # teal
    12: (230, 126, 34),    # dark orange
    13: (142, 68, 173),    # purple
    14: (39, 174, 96),     # dark green
    15: (22, 160, 133),    # dark teal
}

MEMORY_TYPE_COLORS: dict[str, str] = {
    "ACTION": "bright_blue",
    "RULE": "yellow",
    "VOCAB": "bright_green",
    "SUBGOAL": "cyan",
    "PLAN": "bright_magenta",
    "OBSERVATION": "white",
}

MODE_STYLES: dict[str, str] = {
    "LEARN_ACTION": "bold bright_blue",
    "LEARN_SUBGOAL": "bold cyan",
    "LEARN_PLAN": "bold bright_magenta",
    "SOLVE": "bold red",
}

MODE_BORDER: dict[str, str] = {
    "LEARN_ACTION": "bright_blue",
    "LEARN_SUBGOAL": "cyan",
    "LEARN_PLAN": "bright_magenta",
    "SOLVE": "red",
}


# ---------------------------------------------------------------------------
# Step data — everything captured per agent step
# ---------------------------------------------------------------------------
@dataclass
class StepData:
    step: int = 0
    action_name: str = ""
    mode: str = ""
    prediction: str = ""
    surprise: float = 0.0
    learner_changed: bool = False
    learner_output: str = ""
    memory_size_before: int = 0
    memory_size_after: int = 0
    game_state: str = ""
    levels_completed: int = 0
    subgoal: str = ""
    plan: str = ""
    queued: int = 0
    in_subgoal_seq: bool = False
    diagnosis: str = ""


@dataclass
class Captures:
    """Holds the latest LLM prompt/output for each component."""
    curiosity: dict[str, Any] = field(default_factory=dict)
    solver: dict[str, Any] = field(default_factory=dict)
    learner: dict[str, Any] = field(default_factory=dict)
    router: dict[str, Any] = field(default_factory=dict)
    last_surprise: float = 0.0
    _router_context: bool = False


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def render_grid(grid: list[list[int]]) -> Text:
    """Render 64x64 grid using Unicode half-blocks.

    Packs 2 grid rows into 1 terminal row. The top pixel is the foreground
    color and the bottom pixel is the background color of each character.
    """
    if not grid:
        return Text("(no grid)")

    height = len(grid)
    width = len(grid[0]) if grid else 0
    text = Text()

    for y in range(0, height, 2):
        for x in range(width):
            top = grid[y][x] if y < height else 0
            bot = grid[y + 1][x] if y + 1 < height else 0
            r_top, g_top, b_top = ARC_PALETTE.get(top, (128, 128, 128))
            r_bot, g_bot, b_bot = ARC_PALETTE.get(bot, (128, 128, 128))
            style = Style(
                color=Color.from_rgb(r_top, g_top, b_top),
                bgcolor=Color.from_rgb(r_bot, g_bot, b_bot),
            )
            text.append("\u2580", style=style)
        text.append("\n")

    return text


def render_memory(memory: Any) -> Panel:
    """Render memory entries as a colored panel."""
    if not memory or not memory.entries:
        return Panel(
            Text("(empty)", style="dim"),
            title=f"Memory (0/{getattr(memory, 'MAX_ENTRIES', 50)})",
            border_style="blue",
        )

    lines = Text()
    for i, entry in enumerate(memory.entries):
        type_color = MEMORY_TYPE_COLORS.get(entry.type, "white")
        # Index
        lines.append(f"{i:>2} ", style="dim")
        # Type tag
        lines.append(f"[{entry.type:<6s}] ", style=type_color)
        # Content (truncated)
        content = entry.content[:50]
        lines.append(content, style="white")
        # Confidence bar
        bar_len = int(entry.confidence * 10)
        bar = "\u2588" * bar_len + "\u2591" * (10 - bar_len)
        conf_color = "green" if entry.confidence >= 0.7 else "yellow" if entry.confidence >= 0.4 else "red"
        lines.append(f" {bar} ", style=conf_color)
        lines.append(f"{entry.confidence:.2f}", style="dim")
        lines.append("\n")

    return Panel(
        lines,
        title=f"Memory ({len(memory)}/{memory.MAX_ENTRIES})",
        border_style="blue",
    )


def render_status(agent: Any, step_data: StepData) -> Panel:
    """Render the status bar with key agent state."""
    t = Table.grid(expand=True, padding=(0, 2))
    t.add_column(ratio=1)
    t.add_column(ratio=1)
    t.add_column(ratio=1)

    t.add_row(
        Text.assemble(
            ("Action: ", "dim"),
            (step_data.action_name, "bold"),
        ),
        Text.assemble(
            ("Prediction: ", "dim"),
            (f'"{step_data.prediction[:40]}"' if step_data.prediction else "-", "italic"),
        ),
        Text.assemble(
            ("Surprise: ", "dim"),
            (f"{step_data.surprise:.3f}", "bold yellow"),
        ),
    )

    plan_text = step_data.plan[:40] if step_data.plan else "-"
    subgoal_text = step_data.subgoal[:40] if step_data.subgoal else "-"
    learner_text = "CHANGED" if step_data.learner_changed else "NONE"
    learner_style = "bold yellow" if step_data.learner_changed else "dim"

    t.add_row(
        Text.assemble(("Plan: ", "dim"), (plan_text, "")),
        Text.assemble(("Subgoal: ", "dim"), (subgoal_text, "cyan")),
        Text.assemble(("Learner: ", "dim"), (learner_text, learner_style)),
    )

    # Third row: levels, state, diagnosis, sequence status
    win_levels = getattr(agent.frames[-1], "win_levels", "?") if agent.frames else "?"
    diagnosis = step_data.diagnosis
    diag_style = "bold yellow" if diagnosis and diagnosis != "none" else "dim"
    diag_text = diagnosis if diagnosis and diagnosis != "none" else "-"

    t.add_row(
        Text.assemble(
            ("Levels: ", "dim"),
            (f"{step_data.levels_completed}/{win_levels}", "bold"),
            ("  State: ", "dim"),
            (step_data.game_state, ""),
        ),
        Text.assemble(
            ("Diagnosis: ", "dim"),
            (diag_text, diag_style),
            (f"  Queued: {step_data.queued}", "dim"),
        ),
        Text.assemble(
            ("Seq: ", "dim"),
            ("active" if step_data.in_subgoal_seq else "-", "cyan" if step_data.in_subgoal_seq else "dim"),
        ),
    )

    mode = step_data.mode
    mode_style = MODE_STYLES.get(mode, "bold")
    border = MODE_BORDER.get(mode, "bright_black")

    return Panel(
        t,
        title=f"[{mode_style}]{mode}[/]",
        border_style=border,
    )


def render_llm_io(captures: Captures, agent: Any = None) -> Panel:
    """Show the latest LLM interaction for each component.

    Shows full model output (chain of thought + answer) since that's
    the most useful debugging info.
    """
    sections: list[Text] = []

    for name, color in [("curiosity", "cyan"), ("solver", "magenta"), ("learner", "yellow"), ("router", "bright_magenta")]:
        data = getattr(captures, name, {})
        if not data:
            continue

        prompt = data.get("prompt", "")
        output = data.get("output", "")

        header = Text()
        header.append(f"\u2500\u2500 {name.capitalize()} ", style=f"bold {color}")

        # Show key prompt context on the header line
        if prompt:
            ctx_parts = []
            for marker in ["PREDICTION:", "SUBGOAL:", "PLAN:", "CURRENT_MODE:"]:
                for pline in prompt.splitlines():
                    if marker in pline:
                        val = pline.split(marker, 1)[1].strip()[:60]
                        if val and val != "none" and val != "no prediction":
                            ctx_parts.append(f"{marker} {val}")
                        break
            if ctx_parts:
                header.append("(" + ", ".join(ctx_parts) + ")", style="dim")

        sections.append(header)

        # Full model output
        if output:
            for out_line in output.strip().splitlines():
                stripped = out_line.strip()
                if not stripped:
                    continue
                line_text = Text()
                if stripped.upper().startswith("ANSWER"):
                    line_text.append(f"  {stripped}", style=f"bold {color}")
                elif stripped.upper().startswith("EXPECTED"):
                    line_text.append(f"  {stripped}", style="italic")
                elif stripped.upper().startswith(("ADD", "MODIFY", "REMOVE")):
                    line_text.append(f"  {stripped}", style="bold yellow")
                elif stripped.upper() == "NONE":
                    line_text.append(f"  {stripped}", style="dim")
                else:
                    line_text.append(f"  {stripped}", style="dim white")
                sections.append(line_text)
        else:
            sections.append(Text("  (no response)", style="dim"))

    if not sections:
        sections.append(Text("(no LLM calls yet)", style="dim"))

    combined = Text()
    for i, section in enumerate(sections):
        combined.append_text(section)
        combined.append("\n")

    return Panel(combined, title="LLM I/O", border_style="bright_black")


def render_history(history: list[StepData], max_rows: int = 12) -> Panel:
    """Render the step history as a table."""
    table = Table(
        show_header=True, header_style="bold", expand=True, padding=(0, 1),
        border_style="bright_black",
    )
    table.add_column("Step", style="dim", width=4, justify="right")
    table.add_column("Action", width=10)
    table.add_column("Mode", width=14)
    table.add_column("Surpr", width=6, justify="right")
    table.add_column("\u0394Mem", width=5, justify="right")
    table.add_column("Prediction", ratio=1, max_width=80)

    visible = history[-max_rows:]
    for sd in visible:
        mode_style = MODE_STYLES.get(sd.mode, "")
        mem_delta = sd.memory_size_after - sd.memory_size_before
        mem_str = f"+{mem_delta}" if mem_delta > 0 else str(mem_delta) if mem_delta < 0 else "\u00b7"
        mem_style = "yellow" if mem_delta != 0 else "dim"

        table.add_row(
            str(sd.step),
            sd.action_name,
            Text(sd.mode, style=mode_style),
            f"{sd.surprise:.3f}",
            Text(mem_str, style=mem_style),
            Text(sd.prediction if sd.prediction else "-", style="dim italic"),
        )

    return Panel(table, title=f"History ({len(history)} steps)", border_style="bright_black")


def build_header(agent: Any, step: int) -> Text:
    """Build the top header bar."""
    game_id = getattr(agent, "game_id", "?")
    max_actions = getattr(agent, "MAX_ACTIONS", 200)
    mode = agent.current_mode.value if hasattr(agent, "current_mode") else "?"
    mode_style = MODE_STYLES.get(mode, "bold")

    t = Text()
    t.append(f" Game: {game_id} ", style="bold")
    t.append("\u2502", style="dim")
    t.append(f" Step {step}/{max_actions} ", style="bold")
    t.append("\u2502", style="dim")
    t.append(" Mode: ", style="dim")
    t.append(f"{mode} ", style=mode_style)
    return t


# ---------------------------------------------------------------------------
# Instrumentation — capture LLM calls without modifying component code
# ---------------------------------------------------------------------------

def instrument_agent(agent: Any, captures: Captures) -> None:
    """Monkey-patch _call_llm on all LLM-using components to capture I/O.

    Uses a router-context flag so that when curiosity.propose_mode() calls
    _call_llm, the output is routed to captures.router instead of
    captures.curiosity.
    """
    for name in ["curiosity", "solver", "learner"]:
        component = getattr(agent, name, None)
        if component is None or not hasattr(component, "_call_llm"):
            continue
        original_fn = component._call_llm

        def make_wrapper(comp_name: str, orig: Any) -> Any:
            def wrapper(prompt: str, *args: Any, **kwargs: Any) -> str:
                target = "router" if (comp_name == "curiosity" and captures._router_context) else comp_name
                cap = {"prompt": prompt, "output": None}
                setattr(captures, target, cap)
                result = orig(prompt, *args, **kwargs)
                cap["output"] = result
                setattr(captures, target, cap)
                return result
            return wrapper

        component._call_llm = make_wrapper(name, original_fn)

    # Wrap propose_mode to set router context flag
    if hasattr(agent.curiosity, "propose_mode"):
        orig_propose_mode = agent.curiosity.propose_mode

        def mode_wrapper(*args: Any, **kwargs: Any) -> Any:
            captures._router_context = True
            try:
                return orig_propose_mode(*args, **kwargs)
            finally:
                captures._router_context = False

        agent.curiosity.propose_mode = mode_wrapper

    # Capture surprise scores
    if hasattr(agent, "surprise") and hasattr(agent.surprise, "compute"):
        orig_compute = agent.surprise.compute

        def surprise_wrapper(*args: Any, **kwargs: Any) -> float:
            score = orig_compute(*args, **kwargs)
            captures.last_surprise = score
            return score

        agent.surprise.compute = surprise_wrapper


# ---------------------------------------------------------------------------
# Main debug loop
# ---------------------------------------------------------------------------

def run(
    game_id: str,
    auto_delay: Optional[float] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> None:
    """Run the LoopAgent on a game with the Rich TUI debugger."""
    # Set env vars before importing agent
    if base_url:
        os.environ["VLLM_BASE_URL"] = base_url
    if model:
        os.environ["VLLM_MODEL"] = model

    from arc_agi import Arcade

    from agents.templates.loop_agent import LoopAgent

    console.print("\n[bold]Starting LoopAgent debugger[/bold]")
    console.print(f"  Game:  {game_id}")
    console.print(f"  VLLM:  {os.environ.get('VLLM_BASE_URL', 'http://localhost:8000/v1')}")
    console.print(f"  Model: {os.environ.get('VLLM_MODEL', 'google/gemma-3-1b-it')}")
    console.print()

    # Set up arcade + environment
    arcade = Arcade()
    card_id = arcade.open_scorecard(tags=["debug", "loop"])
    env = arcade.make(game_id, scorecard_id=card_id)

    # Create agent
    agent = LoopAgent(
        card_id=card_id,
        game_id=game_id,
        agent_name="loop",
        ROOT_URL="",
        record=False,
        arc_env=env,
    )

    # Instrument
    captures = Captures()
    instrument_agent(agent, captures)

    # History
    history: list[StepData] = []
    auto_mode = auto_delay is not None

    agent.timer = time.time()

    console.print("[dim]Press Enter to step, 'r' for run mode, 'q' to quit[/dim]\n")

    try:
        while (
            not agent.is_done(agent.frames, agent.frames[-1])
            and agent.action_counter <= agent.MAX_ACTIONS
        ):
            # --- Pre-step ---
            latest_frame = agent._convert_raw_frame_data(
                agent.arc_env.observation_space if agent.arc_env else None
            )
            state_before = agent.state_encoder.encode(latest_frame)
            agent._current_state_text = state_before
            grid_before = latest_frame.frame[-1] if latest_frame.frame else []

            mem_size_before = len(agent.memory)
            mode_before = agent.current_mode.value

            # --- Choose + execute ---
            action = agent.choose_action(agent.frames, latest_frame)
            frame_after = agent.take_action(action)

            if frame_after:
                agent.append_frame(frame_after)
                agent._post_step(state_before, grid_before, action, frame_after)

            agent.action_counter += 1

            # --- Collect step data (after _post_step so mode routing has run) ---
            sd = StepData(
                step=agent.action_counter,
                action_name=action.name,
                mode=agent.current_mode.value,
                prediction=agent._last_prediction,
                surprise=captures.last_surprise,
                learner_changed=(len(agent.memory) != mem_size_before),
                learner_output=captures.learner.get("output", "") if captures.learner else "",
                memory_size_before=mem_size_before,
                memory_size_after=len(agent.memory),
                game_state=frame_after.state.name if frame_after else "?",
                levels_completed=frame_after.levels_completed if frame_after else 0,
                subgoal=agent._active_subgoal or "",
                plan=agent._active_plan_text or "",
                queued=len(agent._pending_subgoal_actions),
                in_subgoal_seq=agent._subgoal_sequence_active,
                diagnosis=getattr(agent, "_last_boundary_diagnosis_level", "none"),
            )
            history.append(sd)

            # --- Render ---
            console.clear()

            # Header
            console.print(build_header(agent, sd.step))
            console.print()

            # Grid + Memory side by side
            grid = frame_after.frame[-1] if frame_after and frame_after.frame else (
                latest_frame.frame[-1] if latest_frame.frame else []
            )
            grid_panel = Panel(
                render_grid(grid),
                title="Grid",
                border_style="bright_black",
                width=70,
            )
            memory_panel = render_memory(agent.memory)

            console.print(Columns([grid_panel, memory_panel], expand=True))

            # Status
            console.print(render_status(agent, sd))

            # LLM I/O
            console.print(render_llm_io(captures, agent))

            # History
            console.print(render_history(history))

            # Mode transition indicator
            if sd.mode != mode_before:
                mode_before_style = MODE_STYLES.get(mode_before, "")
                mode_after_style = MODE_STYLES.get(sd.mode, "")
                console.print(
                    Text.assemble(
                        ("  >>> MODE TRANSITION: ", "bold yellow"),
                        (mode_before, mode_before_style),
                        (" \u2192 ", "bold yellow"),
                        (sd.mode, mode_after_style),
                    )
                )

            # Footer
            mode_text = f"[auto {auto_delay}s]" if auto_mode else "[step]"
            console.print(
                f"\n  {mode_text} [dim]Enter=step  r=run  s=step  q=quit[/dim]",
                end="",
            )

            # --- Wait for input ---
            if auto_mode:
                time.sleep(auto_delay)
            else:
                try:
                    user_input = input("  ").strip().lower()
                except EOFError:
                    break

                if user_input == "q":
                    break
                elif user_input == "r":
                    auto_mode = True
                    auto_delay = 0.3
                elif user_input.startswith("r "):
                    auto_mode = True
                    try:
                        auto_delay = float(user_input.split()[1])
                    except (IndexError, ValueError):
                        auto_delay = 0.3
                elif user_input == "s":
                    auto_mode = False

    except KeyboardInterrupt:
        console.print("\n[bold red]Interrupted[/bold red]")
    finally:
        agent.cleanup()
        try:
            arcade.close_scorecard(card_id)
        except Exception:
            pass

    # Final summary
    final_mode = agent.current_mode.value if hasattr(agent, "current_mode") else "?"
    final_mode_style = MODE_STYLES.get(final_mode, "bold")

    console.print("\n")
    console.print(Panel(
        Group(
            Text.assemble(
                ("Game: ", "dim"), (game_id, "bold"), "\n",
                ("Steps: ", "dim"), (str(agent.action_counter), "bold"), "\n",
                ("Final state: ", "dim"),
                (agent.frames[-1].state.name if agent.frames else "?", "bold"), "\n",
                ("Levels completed: ", "dim"),
                (str(agent.frames[-1].levels_completed if agent.frames else 0), "bold"), "\n",
                ("Final mode: ", "dim"), (final_mode, final_mode_style), "\n",
                ("Memory entries: ", "dim"), (str(len(agent.memory)), "bold"), "\n",
            ),
            render_history(history, max_rows=30),
        ),
        title="[bold]Session Summary[/bold]",
        border_style="green",
    ))

    # Print final memory
    if agent.memory.entries:
        console.print(render_memory(agent.memory))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive Rich TUI debugger for the LoopAgent"
    )
    parser.add_argument(
        "--game", "-g",
        required=True,
        help="Game ID to play",
    )
    parser.add_argument(
        "--delay", "-d",
        type=float,
        default=None,
        help="Auto-run delay in seconds (omit for step mode)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="VLLM base URL (default: from env or http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="VLLM model name (default: from env or google/gemma-3-1b-it)",
    )
    args = parser.parse_args()

    run(
        game_id=args.game,
        auto_delay=args.delay,
        base_url=args.base_url,
        model=args.model,
    )


if __name__ == "__main__":
    main()
