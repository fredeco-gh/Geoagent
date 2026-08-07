"""Ensure all user-facing command suggestions say 'geoagent', not 'jutul-agent'.

These tests catch regressions where a framework-level string is updated or
re-introduced with the wrong CLI name.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock, patch

from jutul_agent.workspace import WorkspaceConfig

# ---------------------------------------------------------------------------
# Static scan — backtick-quoted geoagent commands in the files we own.
# Only checks the commands a geoagent user would run: init, key, doctor.
# Developer-only commands (web, tui, sessions, review, upgrade) are left as
# jutul-agent because they have no geoagent equivalent.
# ---------------------------------------------------------------------------

_SRC_ROOT = Path(__file__).parent.parent / "src"

_GEOAGENT_COMMANDS = ["init", "key", "doctor"]

# Files where we replaced jutil-agent suggestions with geoagent.
_USER_FACING_FILES = [
    "jutul_agent/interfaces/cli/doctor.py",
    "jutul_agent/interfaces/cli/init.py",
    "jutul_agent/simulators/env_setup.py",
    "jutul_agent/update_check.py",
    "jutul_agent/agent/plot_julia.py",
]


def test_no_jutul_agent_for_geoagent_commands_in_source() -> None:
    """No backtick-quoted `jutul-agent init/key/doctor` should appear in the
    files we converted to geoagent branding."""
    violations: list[str] = []
    pattern = re.compile(r"`jutul-agent\s+(?:" + "|".join(_GEOAGENT_COMMANDS) + r")\b")
    for rel in _USER_FACING_FILES:
        path = _SRC_ROOT / rel
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith(("#", '"""', "'''")):
                continue
            if pattern.search(line):
                violations.append(f"{rel}:{lineno}: {stripped}")
    assert not violations, (
        "Found `jutul-agent` command suggestions that should say `geoagent`:\n"
        + "\n".join(violations)
    )


# ---------------------------------------------------------------------------
# Doctor output.
# ---------------------------------------------------------------------------


def _run_doctor_mocked(capsys):
    """Run the doctor command with all workspace checks mocked out."""
    from jutul_agent.interfaces.cli import doctor as d

    fake_ws = Path("/fake/workspace")
    fake_config = WorkspaceConfig(simulator="fimbul")

    with (
        patch.object(d, "_check_julia", return_value=MagicMock(ok=True)),
        patch.object(d, "_check_model_and_key"),
        patch.object(d, "_check_simulator", return_value="fimbul"),
        patch.object(d, "_check_julia_project", return_value=fake_ws / "julia-env"),
        patch.object(d, "_check_simulator_installed"),
        patch.object(d, "_check_env_template_current"),
        patch.object(d, "_check_plotting_display"),
        patch.object(d, "_check_julia_runs"),
        patch("jutul_agent.interfaces.cli.doctor.workspace_root", return_value=fake_ws),
        patch("jutul_agent.interfaces.cli.doctor.load_workspace_config", return_value=fake_config),
    ):
        d.run(d.build_parser().parse_args([]))

    return capsys.readouterr().out


def test_doctor_header_uses_geoagent(capsys) -> None:
    """The doctor header line must say 'geoagent doctor', not 'jutul-agent doctor'."""
    out = _run_doctor_mocked(capsys)
    assert "geoagent doctor" in out
    assert "jutul-agent doctor" not in out


def test_doctor_all_passed_message_uses_geoagent(capsys) -> None:
    """The 'all checks passed' summary must reference 'geoagent'."""
    out = _run_doctor_mocked(capsys)
    assert "geoagent" in out
    assert "jutul-agent" not in out


# ---------------------------------------------------------------------------
# env_setup rebuild / retry strings.
# ---------------------------------------------------------------------------


def test_env_setup_rebuild_command_uses_geoagent() -> None:
    """The rebuild/retry command strings in Julia env error messages must say 'geoagent'."""
    import inspect

    import jutul_agent.simulators.env_setup as m

    src = inspect.getsource(m)
    for match in re.finditer(r'\b(?:rebuild|retry)\s*=\s*"([^"]+)"', src):
        cmd = match.group(1)
        assert "jutul-agent" not in cmd, (
            f"env_setup rebuild/retry command still uses 'jutul-agent': {cmd!r}"
        )
        assert "geoagent" in cmd, (
            f"env_setup rebuild/retry command does not use 'geoagent': {cmd!r}"
        )


# ---------------------------------------------------------------------------
# update_check upgrade suggestion.
# ---------------------------------------------------------------------------


def test_update_message_uses_geoagent(monkeypatch, capsys) -> None:
    """The 'newer version available' message must say 'geoagent' and suggest
    the correct uv reinstall command, not 'jutul-agent upgrade'."""
    from jutul_agent import update_check as u
    from jutul_agent.update_check import InstallInfo

    monkeypatch.setattr(u, "install_info", lambda: InstallInfo("registry"))
    monkeypatch.setattr(u, "_refresh_in_background", lambda: None)
    u._write_cache("9.9.9")
    u.notify_at_launch()

    err = capsys.readouterr().err
    assert "geoagent" in err
    assert "jutul-agent" not in err
    assert "uv tool install --reinstall" in err
