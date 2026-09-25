"""List the console commands installed by the mambaglue package.

Run with ``uv run mambaglue-help``. The command descriptions come from the
first line of each module's docstring, read with :mod:`ast` so that no heavy
dependencies (torch, gradio, onnxruntime) are imported just to print help.
"""

from __future__ import annotations

import ast
from importlib import metadata
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent

EXAMPLES = (
    ("uv run mambaglue-camera camera0", "live webcam matching demo"),
    ("uv run mambaglue-demo", "Gradio two-image demo"),
    ("uv run mambaglue-smoke", "released-weights smoke test"),
)


def module_summary(module: str) -> str:
    """Return the first docstring line of ``module`` without importing it."""
    relative = module.removeprefix("mambaglue.").replace(".", "/") + ".py"
    try:
        tree = ast.parse((PACKAGE_DIR / relative).read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return ""
    docstring = (ast.get_docstring(tree) or "").strip()
    return docstring.splitlines()[0] if docstring else ""


def console_commands() -> list[tuple[str, str]]:
    """Return sorted ``(command, summary)`` pairs for console scripts."""
    try:
        distribution = metadata.distribution("mambaglue")
    except metadata.PackageNotFoundError:
        return []
    commands = [
        (entry.name, module_summary(entry.value.split(":", 1)[0]))
        for entry in distribution.entry_points
        if entry.group == "console_scripts"
    ]
    return sorted(commands)


def main() -> None:
    commands = console_commands()
    print("MambaGlue commands (run with: uv run <command>)\n")
    if not commands:
        print("  no installed commands found; run 'uv sync' first")
        return
    width = max(len(name) for name, _ in commands)
    for name, summary in commands:
        line = f"  {name:<{width}}"
        print(f"{line}  {summary}" if summary else line)
    print("\nExamples:")
    for command, note in EXAMPLES:
        print(f"  {command:<35}  {note}")


if __name__ == "__main__":
    main()
