"""Tests for the geothermal-map example's Python wiring (no Julia or browser needed)."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import re
from pathlib import Path
from typing import Any

from fakes import FakeJulia, make_fake_adapter
from jutul_agent.juliakernel.result import EvalResult
from jutul_agent.session import Session

_CAPABILITY_PATH = Path(__file__).resolve().parents[1] / "geoagent_app" / "capability.py"


def _load_capability():
    spec = importlib.util.spec_from_file_location(
        "geothermal_map_capability_module", _CAPABILITY_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


capability = _load_capability()

_FEATURES = [
    {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [10.7, 59.9]},
        "properties": {"layer": "EnergiBrønn", "brønnNr": "100", "beskrivelse": "Test well"},
    },
    {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [10.8, 60.0]},
        "properties": {"layer": "BrønnPark", "brønnParkNr": "200", "brønnpOmrNavn": "Test park"},
    },
]


def _session(tmp_path: Path, julia: FakeJulia | None = None) -> Session:
    return Session.create(
        julia=julia or FakeJulia(), simulator=make_fake_adapter(tmp_path), state_root=tmp_path
    )


def _data_path(tmp_path: Path) -> str:
    path = tmp_path / "boreholes.geojson"
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": _FEATURES}), encoding="utf-8"
    )
    return str(path)


def test_capability_is_web_with_five_tools() -> None:
    cap = capability.geothermal_map_capability("unused", "unused")
    assert cap.surfaces == ("web",)
    assert len(cap.tools) == 16
    assert cap.prompt_fragment
    # Auto-pins the map the moment a session connects (Phase 4), rather than
    # only once some map tool happens to run.
    assert cap.on_connect == (capability._ensure_map_pinned,)


def test_ensure_map_pinned_appends_a_silent_map_viz(tmp_path: Path) -> None:
    session = _session(tmp_path)
    capability._ensure_map_pinned(session)
    artifacts = [e for e in session.trace.iter_events() if e.kind == "artifact"]
    assert len(artifacts) == 1
    assert artifacts[-1].payload["kind"] == "map"
    # Not a conversation event worth a chat reference (see protocol.viz_to_wire).
    assert artifacts[-1].payload["silent"] is True


def test_ensure_map_pinned_is_idempotent(tmp_path: Path) -> None:
    session = _session(tmp_path)
    capability._ensure_map_pinned(session)
    capability._ensure_map_pinned(session)
    artifacts = [e for e in session.trace.iter_events() if e.kind == "artifact"]
    assert len(artifacts) == 1  # the second call is a no-op, not a re-pin


def test_set_map_view_emits_a_targeted_ui_action(tmp_path: Path) -> None:
    session = _session(tmp_path)
    set_map_view = capability._make_set_map_view_tool(session)
    out = asyncio.run(set_map_view.ainvoke({"lon": 10.5, "lat": 59.9, "zoom": 12}))
    assert "10.5" in out
    ui_events = [e for e in session.trace.iter_events() if e.kind == "ui"]
    assert ui_events[-1].payload == {
        "action": "set_map_view",
        "payload": {"lon": 10.5, "lat": 59.9, "zoom": 12.0},
        "target": "slot:geothermal-map",
    }


def test_go_to_well_finds_an_exact_number_match(tmp_path: Path) -> None:
    session = _session(tmp_path)
    go_to_well = capability._make_go_to_well_tool(session, _data_path(tmp_path))
    out = asyncio.run(go_to_well.ainvoke({"identifier": "100"}))
    assert "100" in out
    ui_events = [e for e in session.trace.iter_events() if e.kind == "ui"]
    payload = ui_events[-1].payload
    assert payload["action"] == "go_to_well"
    assert payload["target"] == "slot:geothermal-map"
    assert payload["payload"]["lon"] == 10.7
    assert payload["payload"]["lat"] == 59.9
    assert payload["payload"]["feature"]["properties"]["brønnNr"] == "100"


def test_go_to_well_reports_not_found_without_claiming_success(tmp_path: Path) -> None:
    session = _session(tmp_path)
    go_to_well = capability._make_go_to_well_tool(session, _data_path(tmp_path))
    out = asyncio.run(go_to_well.ainvoke({"identifier": "no-such-well"}))
    assert "No well matching" in out
    assert not [e for e in session.trace.iter_events() if e.kind == "ui"]


def test_go_to_well_park_only_matches_park_features(tmp_path: Path) -> None:
    session = _session(tmp_path)
    go_to_well_park = capability._make_go_to_well_park_tool(session, _data_path(tmp_path))
    out = asyncio.run(go_to_well_park.ainvoke({"identifier": "200"}))
    assert "200" in out
    ui_events = [e for e in session.trace.iter_events() if e.kind == "ui"]
    feature = ui_events[-1].payload["payload"]["feature"]
    assert feature["properties"]["layer"] == "BrønnPark"

    # A well sharing the park's number (none here) must not satisfy the park
    # lookup — only the park's own number/area fields qualify (_PARK_*_FIELDS).
    out2 = asyncio.run(go_to_well_park.ainvoke({"identifier": "100"}))
    assert "No well park matching" in out2


def test_map_is_pinned_once_per_session(tmp_path: Path) -> None:
    session = _session(tmp_path)
    set_map_view = capability._make_set_map_view_tool(session)
    asyncio.run(set_map_view.ainvoke({"lon": 1, "lat": 2}))
    asyncio.run(set_map_view.ainvoke({"lon": 3, "lat": 4}))
    artifacts = [e for e in session.trace.iter_events() if e.kind == "artifact"]
    assert len(artifacts) == 1
    assert artifacts[0].payload["kind"] == "map"
    assert artifacts[0].payload["slot"] == "geothermal-map"


# ---------------------------------------------------------------------------
# Simulation (Phase 2): FakeJulia never runs real Julia, so each handler below
# stands in for simulation.jl itself — it locates the result-file path the
# *rendered* template would have told Julia to write to (via the same
# raw"...".json/.b64 literals _render_template substitutes in), and writes a
# synthetic result there directly, exactly as if simulation.jl had run.


def _result_path(code: str) -> Path:
    match = re.search(r'raw"([^"]+\.json)"', code)
    assert match, f"no .json result path found in rendered code:\n{code}"
    return Path(match.group(1))


def _b64_path(code: str) -> Path:
    match = re.search(r'raw"([^"]+\.b64)"', code)
    assert match, f"no .b64 result path found in rendered code:\n{code}"
    return Path(match.group(1))


def _simulate_eval_handler(result: dict[str, Any]):
    def handler(code: str) -> EvalResult:
        assert "run_fimbul_simulation" in code
        _result_path(code).write_text(json.dumps(result), encoding="utf-8")
        return EvalResult(output="ok")

    return handler


def _setup_eval_handler(result: dict[str, Any]):
    def handler(code: str) -> EvalResult:
        assert "well_to_simulation_params" in code
        _result_path(code).write_text(json.dumps(result), encoding="utf-8")
        return EvalResult(output="ok")

    return handler


def _view_eval_handler(b64: str):
    def handler(code: str) -> EvalResult:
        assert "render_reservoir_image" in code or b64 == ""
        _b64_path(code).write_text(b64, encoding="ascii")
        return EvalResult(output="ok")

    return handler


_SIM_RESULT = {
    "status": "completed",
    "message": "Simulation completed successfully.",
    "well_data": {"Well1": {"Temperature": [10.0, 12.0]}},
    "timestamps": [0.0, 365.0],
    "num_steps": 2,
    "reservoir_vars": [],
}


def test_run_simulation_tool_records_report_and_returns_summary(tmp_path: Path) -> None:
    julia = FakeJulia(eval_handler=_simulate_eval_handler(_SIM_RESULT))
    session = _session(tmp_path, julia)
    action = capability.make_run_simulation_action("unused.jl")
    sent: list[dict[str, Any]] = []

    async def _run() -> None:
        async def send_wire(msg: dict[str, Any]) -> None:
            sent.append(msg)

        await action(
            session,
            {"case_type": "AGS", "parameters": {"well_length": 100.0}},
            send_wire,
            lambda _: None,
        )

    asyncio.run(_run())
    finished = next(m for m in sent if m.get("type") == "tool" and m.get("event") == "finished")
    assert "AGS" in finished["content"]
    assert "Well1" in finished["content"]
    artifacts = [e for e in session.trace.iter_events() if e.kind == "artifact"]
    assert artifacts[-1].payload["path"] == "artifacts/simulation-results-1.html"
    assert artifacts[-1].payload["mime"] == "text/html"


def test_run_simulation_tool_opens_a_new_tab_each_run(tmp_path: Path) -> None:
    julia = FakeJulia(eval_handler=_simulate_eval_handler(_SIM_RESULT))
    session = _session(tmp_path, julia)
    action = capability.make_run_simulation_action("unused.jl")
    args = {"case_type": "AGS", "parameters": {"well_length": 100.0}}

    async def _run() -> None:
        async def send_wire(_: dict[str, Any]) -> None:
            pass

        for _ in range(2):
            await action(session, args, send_wire, lambda _: None)

    asyncio.run(_run())
    artifacts = [e for e in session.trace.iter_events() if e.kind == "artifact"]
    assert len(artifacts) == 2
    # Distinct paths and slots: the second run must open its own tab, not
    # replace the first run's.
    paths = {a.payload["path"] for a in artifacts}
    slots = {a.payload["slot"] for a in artifacts}
    assert len(paths) == 2
    assert len(slots) == 2


def test_run_simulation_tool_reports_failure_without_recording_artifact(tmp_path: Path) -> None:
    failure = {"status": "error", "message": "Invalid parameters: well_length must be positive"}
    julia = FakeJulia(eval_handler=_simulate_eval_handler(failure))
    session = _session(tmp_path, julia)
    action = capability.make_run_simulation_action("unused.jl")
    sent: list[dict[str, Any]] = []

    async def _run() -> None:
        async def send_wire(msg: dict[str, Any]) -> None:
            sent.append(msg)

        await action(session, {"case_type": "AGS", "parameters": {}}, send_wire, lambda _: None)

    asyncio.run(_run())
    error_msg = next(m for m in sent if m.get("type") == "tool" and m.get("event") == "error")
    assert "well_length" in error_msg["content"]
    assert not [e for e in session.trace.iter_events() if e.kind == "artifact"]


def test_view_simulation_result_tool_returns_an_image(tmp_path: Path) -> None:
    julia = FakeJulia(eval_handler=_view_eval_handler("ZmFrZQ=="))
    session = _session(tmp_path, julia)
    view_tool = capability._make_view_simulation_result_tool(session, "unused.jl")
    out = asyncio.run(view_tool.ainvoke({"var": "Temperature", "step": -1, "delta": False}))
    assert isinstance(out, list)
    assert out[1]["type"] == "image"
    assert out[1]["base64"] == "ZmFrZQ=="


def test_view_simulation_result_tool_reports_no_result_yet(tmp_path: Path) -> None:
    julia = FakeJulia(eval_handler=_view_eval_handler(""))
    session = _session(tmp_path, julia)
    view_tool = capability._make_view_simulation_result_tool(session, "unused.jl")
    out = asyncio.run(view_tool.ainvoke({"var": "Temperature", "step": -1, "delta": False}))
    assert "No simulation result" in out


def test_setup_simulation_action_sends_a_targeted_ui_action(tmp_path: Path) -> None:
    params_result = {
        "case_type": "AGS",
        "parameters": {"well_length": {"value": 100.0, "source": "data"}},
    }
    julia = FakeJulia(eval_handler=_setup_eval_handler(params_result))
    session = _session(tmp_path, julia)
    action = capability.make_setup_simulation_action("unused.jl")
    sent: list[dict[str, Any]] = []

    async def send_wire(msg: dict[str, Any]) -> None:
        sent.append(msg)

    asyncio.run(
        action(session, {"layer": "EnergiBrønn", "brønnNr": "100"}, send_wire, lambda _: None)
    )

    assert sent == [
        {
            "type": "ui",
            "action": "simulation_params",
            "payload": params_result,
            "target": "slot:geothermal-map",
        }
    ]


def test_setup_simulation_action_reports_failure_as_a_targeted_ui_action(tmp_path: Path) -> None:
    def handler(code: str) -> EvalResult:
        return EvalResult(output="", error="boom")

    julia = FakeJulia(eval_handler=handler)
    session = _session(tmp_path, julia)
    action = capability.make_setup_simulation_action("unused.jl")
    sent: list[dict[str, Any]] = []

    async def send_wire(msg: dict[str, Any]) -> None:
        sent.append(msg)

    asyncio.run(action(session, {}, send_wire, lambda _: None))

    assert sent[-1]["action"] == "simulation_setup_error"
    assert sent[-1]["target"] == "slot:geothermal-map"
    assert "boom" in sent[-1]["payload"]["message"]


def test_run_simulation_action_streams_tool_events_and_queues_a_ui_event(tmp_path: Path) -> None:
    julia = FakeJulia(
        eval_handler=_simulate_eval_handler(_SIM_RESULT), stream_chunks=["progress...\n"]
    )
    session = _session(tmp_path, julia)
    action = capability.make_run_simulation_action("unused.jl")
    sent: list[dict[str, Any]] = []
    queued: list[Any] = []

    async def send_wire(msg: dict[str, Any]) -> None:
        sent.append(msg)

    asyncio.run(
        action(
            session,
            {"case_type": "AGS", "parameters": {"well_length": 100.0}},
            send_wire,
            queued.append,
        )
    )

    tool_events = [m["event"] for m in sent if m["type"] == "tool"]
    assert tool_events == ["started", "delta", "finished"]
    assert any(m["type"] == "viz" for m in sent)  # the report artifact, re-flushed
    assert len(queued) == 1
    assert queued[0]["event"] == "simulationCompleted"
    assert queued[0]["case_type"] == "AGS"


def test_run_simulation_action_reports_an_error_event_on_failure(tmp_path: Path) -> None:
    failure = {"status": "error", "message": "Invalid parameters: well_length must be positive"}
    julia = FakeJulia(eval_handler=_simulate_eval_handler(failure))
    session = _session(tmp_path, julia)
    action = capability.make_run_simulation_action("unused.jl")
    sent: list[dict[str, Any]] = []
    queued: list[Any] = []

    async def send_wire(msg: dict[str, Any]) -> None:
        sent.append(msg)

    asyncio.run(action(session, {"case_type": "AGS", "parameters": {}}, send_wire, queued.append))

    tool_events = [m["event"] for m in sent if m["type"] == "tool"]
    assert tool_events == ["started", "error"]
    assert not queued
    assert not [e for e in session.trace.iter_events() if e.kind == "artifact"]
