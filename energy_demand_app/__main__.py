"""geoagent — CLI entry point for the Geoagent geothermal planning tool."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_WORKSPACE = Path.home() / ".geoagent" / "workspace"

_USAGE = """\
Usage: geoagent [command]

Commands:
  key     Save an LLM provider API key (e.g. geoagent key openai).
  init    Precompile Julia/Fimbul (~10 min, one-time). Run before the first serve.
  serve   Start the Geoagent web server (default if no command given).
"""


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "serve"

    if cmd == "serve":
        from energy_demand_app.serve import main as serve_main

        return serve_main()

    if cmd == "init":
        _WORKSPACE.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["jutul-agent", "init", "--sim", "fimbul"],
            cwd=_WORKSPACE,
        )
        return result.returncode

    if cmd == "key":
        result = subprocess.run(["jutul-agent", "key", *sys.argv[2:]])
        return result.returncode

    print(f"Unknown command: {cmd!r}\n{_USAGE}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
