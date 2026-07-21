"""The geothermal map capability: tools that let the agent drive the native map
panel (see ``canvas/MapPanel.tsx`` in jutul-agent's webapp) by well name or
location.

Each tool below emits a `ui` trace event targeted at the map's own canvas view
(``target="slot:geothermal-map"``), which ``MapPanel.tsx`` picks up via its
``useUiActions`` hook — the same agent-to-UI pattern Phase 0's throwaway demo
proved (``fly_test_map``), just for a real well lookup instead of bare
coordinates. There is no dedicated Julia kernel and no second server here: the
map panel lives in the same process and the same session as the chat, ported
from geothermal-viz's ``web/js/app.js`` (rendering) and
``examples/geothermal-viz-app/capability.py`` (this file's predecessor).

This module is the skeleton for growing the agent's reach into the map: adding
a new ability means adding one ``_make_..._tool`` factory below, wiring it into
``geothermal_map_capability``'s ``tools`` tuple, and adding the matching case to
``MapPanel.tsx``'s `ui` action dispatch. Nothing else needs to change to pick
it up — serve.py passes whatever this returns straight into the session.

A tool that *asks* the map to do something (like ``set_map_view``) can return
right away — there's nothing to get wrong. But ``go_to_well``/``go_to_well_park``
resolve the well themselves, against the same GeoJSON file the map renders
from (read directly off disk here — no HTTP self-call, since the tool and the
data live in the same process now), so the agent can give an honest answer
immediately instead of waiting for the browser to report back on the *next*
message (see docs/server-interface.md's ui_event queueing).

The simulation tools (``run_simulation``, ``view_simulation_result``) reuse
geothermal-viz's own ``julia/simulation.jl`` verbatim — ``include()``d once into
the session's own kernel — rather than porting its Fimbul/parameter-mapping
logic into Python. ``make_run_simulation_action``/``make_setup_simulation_action``
are the map's two direct, non-LLM entry points (the sidebar's "Setup
Simulation"/"Run" buttons): for when the front end already has exact,
structured inputs and there is nothing for the model to decide — see
docs/server-interface.md and jutul_agent.interfaces.server.app.ActionHandler.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from jutul_agent.agent.capabilities import Capability
from jutul_agent.juliakernel.result import OutputChunk
from jutul_agent.session import Session

# The map panel's fixed canvas slot: stable across every tool call in a
# session, so re-pinning (see _ensure_map_pinned) refreshes the same view
# instead of stacking a new tab, and `target` always reaches the same panel.
_MAP_SLOT = "geothermal-map"
_MAP_TARGET = f"slot:{_MAP_SLOT}"

# Matched first, exactly (case-insensitive): a well or well-park number.
_EXACT_FIELDS = ("brønnNr", "brønnParkNr")
# Matched next, as a substring, so a vaguer request still has a chance to resolve.
_LOOSE_FIELDS = (*_EXACT_FIELDS, "brønnpOmrNavn", "beskrivelse", "oppdragstaker")

# Same idea, but restricted to the well-park identifier itself — excludes
# `brønnNr` so a well-park lookup can't accidentally land on an unrelated well
# whose own number happens to match the park identifier given.
_PARK_EXACT_FIELDS = ("brønnParkNr",)
_PARK_LOOSE_FIELDS = ("brønnParkNr", "brønnpOmrNavn", "beskrivelse", "oppdragstaker")

# Cached per data path rather than per call: the dataset only changes when the
# offline scripts/process_data.jl reruns, so re-reading the (multi-MB) file on
# every tool call would be wasted work.
_wells_cache: dict[str, list[dict[str, Any]]] = {}

# Sessions that already have the map panel pinned. A fixed slot makes
# re-pinning idempotent in the canvas itself, but the store still appends a
# fresh chip to the thread on every viz message regardless — this guards
# against spamming one per tool call within the same session.
_map_pinned: set[str] = set()

# Per-session count of simulation runs recorded as report tabs (see
# _record_simulation_artifact): each run gets its own number so a later run
# opens an additional tab instead of overwriting the previous run's.
_simulation_run_count: dict[str, int] = {}

# Last successful borehole simulation result per session — consumed by the
# plot tools without re-running the simulation.
_session_borehole_results: dict[str, dict] = {}

# Per-session count of borehole simulation runs; each run gets its own slot so
# plot tabs accumulate instead of overwriting each other.
_borehole_sim_run_count: dict[str, int] = {}

# Last successful Fimbul validation result per session — consumed by the
# view_fimbul_validation plot tool without re-running the simulation.
_session_fimbul_validation_results: dict[str, dict] = {}

# Per-session count of Fimbul validation runs; each run gets its own slot so
# validation tabs accumulate instead of overwriting each other.
_fimbul_validation_run_count: dict[str, int] = {}


def _aggregate_tin_profile(
    T_in: list[float],
    window_hours: int = 24,
) -> tuple[list[float], list[float]]:
    """Average hourly T_in into fixed-width windows.

    The last window may be shorter than window_hours if len(T_in) is not
    divisible by window_hours.

    Returns:
        T_in_avg:    mean T_in per window [°C]
        durations_s: window duration [s]
    """
    n = len(T_in)
    T_in_avg, durations_s = [], []
    i = 0
    while i < n:
        end = min(i + window_hours, n)
        chunk = T_in[i:end]
        T_in_avg.append(sum(chunk) / len(chunk))
        durations_s.append(float(len(chunk) * 3600))
        i = end
    return T_in_avg, durations_s


def _apply_overrides(boreholes, overrides):
    """Apply per-borehole overrides to a generated field.

    overrides: {sector_str: {well_str: {param: value, ...}}}
    dx/dy are added to the borehole's existing x/y; all other keys replace.
    Returns a new nested list; the originals are not mutated.
    """
    if not overrides:
        return boreholes
    import dataclasses
    boreholes = [list(sector) for sector in boreholes]
    for s_str, wells in overrides.items():
        s = int(s_str)
        for w_str, changes in wells.items():
            w = int(w_str)
            b = boreholes[s][w]
            ch = {k: float(v) for k, v in changes.items() if k not in ("dx", "dy")}
            if "dx" in changes:
                ch["x"] = b.x + float(changes["dx"])
            if "dy" in changes:
                ch["y"] = b.y + float(changes["dy"])
            boreholes[s][w] = dataclasses.replace(b, **ch)
    return boreholes


def _load_well_features(data_path: str) -> list[dict[str, Any]]:
    cached = _wells_cache.get(data_path)
    if cached is not None:
        return cached
    geojson = json.loads(Path(data_path).read_text(encoding="utf-8"))
    features = geojson.get("features", [])
    _wells_cache[data_path] = features
    return features


def _find_well(
    features: list[dict[str, Any]],
    identifier: str,
    *,
    exact_fields: tuple[str, ...] = _EXACT_FIELDS,
    loose_fields: tuple[str, ...] = _LOOSE_FIELDS,
) -> dict[str, Any] | None:
    needle = str(identifier).strip().lower()
    if not needle:
        return None
    for feature in features:
        props = feature.get("properties", {})
        if any(str(props.get(f, "")).lower() == needle for f in exact_fields):
            return feature
    for feature in features:
        props = feature.get("properties", {})
        if any(needle in str(props.get(f, "")).lower() for f in loose_fields):
            return feature
    return None


def _ensure_map_pinned(session: Session) -> None:
    """Pin the map panel into the canvas, idempotently.

    Registered as the capability's ``on_connect`` hook (see
    ``geothermal_map_capability``), so the map is there the moment the session's
    WebSocket connects — like geothermal-viz's own always-visible map — rather
    than appearing only once some tool happens to touch it, or only after the
    first prompt. Map tools also call this themselves, a harmless no-op once the
    hook has already run; it only does real work the very first time, for a
    session resuming on a server that predates this hook (e.g. across an
    upgrade). The file written here is never read by MapPanel itself (it
    renders natively, not in an iframe) — it only exists to satisfy the
    artifact plumbing that produces the `viz` wire message.
    """
    if session.session_id in _map_pinned:
        return
    _map_pinned.add(session.session_id)
    rel = "artifacts/geothermal-map.html"
    path = session.output_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("<!doctype html><title>Geothermal map</title>", encoding="utf-8")
    session.trace.append(
        "artifact",
        {
            "path": rel,
            "mime": "text/html",
            "kind": "map",
            "slot": _MAP_SLOT,
            "caption": "Geothermal map",
            # Pinning the map isn't a conversation event worth a chat
            # reference, whether it happens proactively (on_connect) or
            # reactively (a map tool's own call, for an older resumed session).
            "silent": True,
        },
    )


def _make_set_map_view_tool(session: Session):
    @tool
    async def set_map_view(lon: float, lat: float, zoom: float = 14.0) -> str:
        """Move the geothermal map to a location.

        Args:
            lon: Longitude in degrees (WGS84).
            lat: Latitude in degrees (WGS84).
            zoom: Map zoom level — roughly 0 for the whole world, 18 for street
                level. Defaults to 14, which frames a single borehole site.
        """
        _ensure_map_pinned(session)
        session.trace.append(
            "ui",
            {
                "action": "set_map_view",
                "payload": {"lon": lon, "lat": lat, "zoom": zoom},
                "target": _MAP_TARGET,
            },
        )
        return f"Moved the map to ({lat}, {lon}) at zoom {zoom}."

    return set_map_view


def _make_go_to_address_tool(session: Session):
    @tool
    async def go_to_address(address: str, zoom: float = 16.0) -> str:
        """Fly the geothermal map to a Norwegian street address.

        Resolves the address via Kartverket's address registry (Matrikkelen)
        and moves the map to the matching location.  Use this whenever the user
        gives a street address instead of well-number or coordinates.

        Args:
            address: Norwegian street address, e.g. "Karl Johans gate 1, Oslo".
            zoom: Map zoom level after flying there. Default 16 (street level).
        """
        import httpx

        url = "https://ws.geonorge.no/adresser/v1/sok"
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(url, params={"sok": address})
            response.raise_for_status()
            data = response.json()

        adresser = data.get("adresser", [])
        if not adresser:
            return f"No address found for '{address}'. Try a more specific query."

        hit = adresser[0]
        punkt = hit["representasjonspunkt"]
        lat = punkt["lat"]
        lon = punkt["lon"]
        resolved = ", ".join(filter(None, [
            hit.get("adressetekst"),
            hit.get("postnummer"),
            hit.get("poststed"),
            hit.get("kommunenavn"),
        ]))

        _ensure_map_pinned(session)
        session.trace.append(
            "ui",
            {
                "action": "set_map_view",
                "payload": {"lon": lon, "lat": lat, "zoom": zoom},
                "target": _MAP_TARGET,
            },
        )
        return f"Moved the map to {resolved} ({lat:.5f}, {lon:.5f})."

    return go_to_address


def _make_go_to_well_tool(session: Session, data_path: str):
    @tool
    async def go_to_well(identifier: str) -> str:
        """Fly the map to a specific well and select it, as if the user clicked it.

        Args:
            identifier: A well or well-park number (e.g. "12345"), or other
                identifying text (area name, contractor, description) to match
                loosely if no well/park number matches exactly.
        """
        try:
            features = _load_well_features(data_path)
        except Exception as exc:
            return f"Could not read the borehole data to look up wells: {exc}"
        feature = _find_well(features, identifier)
        if feature is None:
            return (
                f"No well matching '{identifier}' was found in the loaded borehole "
                "data — tell the user that."
            )
        lon, lat = feature["geometry"]["coordinates"]
        _ensure_map_pinned(session)
        session.trace.append(
            "ui",
            {
                "action": "go_to_well",
                "payload": {"lon": lon, "lat": lat, "feature": feature},
                "target": _MAP_TARGET,
            },
        )
        return f"Found well '{identifier}' and moved the map to it."

    return go_to_well


def _make_go_to_well_park_tool(session: Session, data_path: str):
    @tool
    async def go_to_well_park(identifier: str) -> str:
        """Fly the map to a well park itself and select it, as if the user
        clicked it directly — use this when asked about a well *park* rather
        than one of the individual wells inside it.

        Args:
            identifier: A well-park number (e.g. "12345"), or other identifying
                text (area name, contractor, description) to match loosely if
                no park number matches exactly.
        """
        try:
            features = _load_well_features(data_path)
        except Exception as exc:
            return f"Could not read the borehole data to look up well parks: {exc}"
        # Well parks are their own feature ("layer" == "BrønnPark"), with their
        # own coordinates — distinct from the individual wells that merely
        # reference one via `brønnParkNr`. Restricting the search to that
        # layer is what actually lands on the park itself; without it, an
        # ordinary well sharing the same park number (the data lists those
        # before any park feature) would match first instead.
        parks = [f for f in features if f.get("properties", {}).get("layer") == "BrønnPark"]
        feature = _find_well(
            parks,
            identifier,
            exact_fields=_PARK_EXACT_FIELDS,
            loose_fields=_PARK_LOOSE_FIELDS,
        )
        if feature is None:
            return (
                f"No well park matching '{identifier}' was found in the loaded "
                "borehole data — tell the user it doesn't exist rather than "
                "saying you moved the map."
            )
        lon, lat = feature["geometry"]["coordinates"]
        _ensure_map_pinned(session)
        # Reuses the same go_to_well ui action: the map only ever flies to and
        # selects one feature regardless of whether it's a well or a well park,
        # so no new action/dispatch case is needed.
        session.trace.append(
            "ui",
            {
                "action": "go_to_well",
                "payload": {"lon": lon, "lat": lat, "feature": feature},
                "target": _MAP_TARGET,
            },
        )
        return f"Found well park '{identifier}' and moved the map to it."

    return go_to_well_park


# ---------------------------------------------------------------------------
# Fimbul simulation, run on the session's own Julia kernel.
#
# simulation.jl is include()d once per kernel (the isdefined guard makes it
# idempotent) and called directly — its functions, caches, and globals
# (_sim_case, _sim_states, render_reservoir_image, ...) are reused exactly as
# geothermal-viz's own server used them, just executed in a different process.
#
# The structured result comes back through a JSON file rather than parsed from
# stdout: simulation.jl's progress lines and Fimbul's own progress-bar output
# share that stream, so splitting a "real" return value out of it by text
# parsing would be fragile. A file gives a clean, unambiguous result.

_SIMULATE_TEMPLATE = '''
begin
    if !isdefined(Main, :run_fimbul_simulation)
        include(raw"__JL_PATH__")
    end
    import JSON3
    local _setup = Dict{String,Any}(
        "case_type" => "__CASE_TYPE__",
        "parameters" => Dict{String,Any}(
        JSON3.read(raw"""__PARAMS_JSON__""", Dict{String,Float64})),
    )
    local _errors = validate_simulation_params(_setup)
    local _result = if !isempty(_errors)
        Dict{String,Any}(
            "status" => "error",
            "message" => "Invalid parameters: " * join(["$(e[1]): $(e[2])" for e in _errors], "; "),
        )
    else
        run_fimbul_simulation(_setup)
    end
    if get(_result, "status", "") == "completed"
        try
            # The HTML report draws its own well-output chart client-side from
            # well_data/timestamps below, so only the reservoir-state images need
            # rendering here. Those need Fimbul's mesh + render_reservoir_image, so
            # they can only happen here, in-kernel — the report can't fetch them
            # lazily afterwards since geothermal-viz no longer serves that. A bounded,
            # evenly-spaced subset of steps is pre-rendered (not every step) to keep
            # this from adding minutes to every run.
            local _rvars = get(_result, "reservoir_vars", String[])
            local _n = get(_result, "num_steps", 0)
            if !isempty(_rvars) && _n > 0
                local _maxshots = 15
                local _steps = _n <= _maxshots ? collect(1:_n) :
                    sort(collect(Set(round.(Int, range(1, _n; length=_maxshots)))))
                local _images = Dict{String,Any}()
                for _v in _rvars
                    local _bystep = Dict{String,Any}()
                    for _s in _steps
                        local _byd = Dict{String,Any}()
                        for _d in (false, true)
                            local _img = render_reservoir_image(_v, _s; delta=_d)
                            if !isempty(_img)
                                _byd[_d ? "true" : "false"] = _img
                            end
                        end
                        if !isempty(_byd)
                            _bystep[string(_s)] = _byd
                        end
                    end
                    _images[_v] = _bystep
                end
                _result["reservoir_images"] = _images
                _result["reservoir_steps"] = _steps
            end
        catch e
            @warn "Could not pre-render reservoir images" exception=e
        end
    end
    open(raw"__RESULT_PATH__", "w") do io
        JSON3.write(io, _result)
    end
    "ok"
end
'''

_VIEW_RESULT_TEMPLATE = """
begin
    if !isdefined(Main, :render_reservoir_image)
        include(raw"__JL_PATH__")
    end
    local _states = _sim_states[]
    if _states === nothing || isempty(_states)
        write(raw"__IMG_PATH__", "")
    else
        local _step = __STEP__ < 0 ? length(_states) : __STEP__
        local _img = render_reservoir_image("__VAR__", _step; delta=__DELTA__)
        write(raw"__IMG_PATH__", _img)
    end
    "ok"
end
"""

_SETUP_PARAMS_TEMPLATE = '''
begin
    if !isdefined(Main, :well_to_simulation_params)
        include(raw"__JL_PATH__")
    end
    import JSON3
    local _props = Dict{String,Any}(JSON3.read(raw"""__PROPS_JSON__""", Dict{String,Any}))
    local _result = well_to_simulation_params(_props)
    open(raw"__RESULT_PATH__", "w") do io
        JSON3.write(io, _result)
    end
    "ok"
end
'''

_BTES_VALIDATION_TEMPLATE = '''
begin
    if !isdefined(Main, :run_btes_validation)
        include(raw"__JL_PATH__")
    end
    import JSON3
    local _setup = JSON3.read(raw"""__SETUP_JSON__""", Dict{String,Any})
    local _result = run_btes_validation(_setup)
    open(raw"__RESULT_PATH__", "w") do io
        JSON3.write(io, _result)
    end
    "ok"
end
'''


def _render_template(template: str, **values: str) -> str:
    code = template
    for key, value in values.items():
        code = code.replace(f"__{key}__", value)
    return code


async def _execute_fimbul_simulation(
    session: Session,
    simulation_jl_path: str,
    case_type: str,
    parameters: dict[str, Any],
    *,
    on_chunk: Callable[[OutputChunk], None] | None = None,
) -> dict[str, Any]:
    """Run a Fimbul simulation in the session's own persistent Julia kernel.

    Validates first (mirroring what geothermal-viz's own server used to do
    before running), then calls simulation.jl's existing run_fimbul_simulation
    unchanged. ``on_chunk``, if given, receives the live progress output exactly
    as ``run_julia`` does for any other long Julia computation.
    """
    result_path = session.output_dir / "artifacts" / f"sim-result-{uuid.uuid4().hex[:8]}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    jl_path = Path(simulation_jl_path).resolve().as_posix()
    code = _render_template(
        _SIMULATE_TEMPLATE,
        JL_PATH=jl_path,
        CASE_TYPE=case_type,
        PARAMS_JSON=json.dumps({k: float(v) for k, v in parameters.items()}),
        RESULT_PATH=result_path.as_posix(),
    )
    result = await session.julia.eval(code, on_chunk=on_chunk)
    if result.error:
        raise RuntimeError(result.error)
    return json.loads(result_path.read_text(encoding="utf-8"))


async def _render_simulation_view(
    session: Session, simulation_jl_path: str, var: str, step: int, delta: bool
) -> str:
    """The base64 PNG for one reservoir-state step of the most recent simulation
    still held in the kernel ("" if there isn't one)."""
    img_path = session.output_dir / "artifacts" / f"sim-view-{uuid.uuid4().hex[:8]}.b64"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    code = _render_template(
        _VIEW_RESULT_TEMPLATE,
        JL_PATH=Path(simulation_jl_path).resolve().as_posix(),
        VAR=var,
        STEP=str(int(step)),
        DELTA="true" if delta else "false",
        IMG_PATH=img_path.as_posix(),
    )
    result = await session.julia.eval(code)
    if result.error:
        raise RuntimeError(result.error)
    return img_path.read_text(encoding="ascii")


async def _setup_simulation_params(
    session: Session, simulation_jl_path: str, properties: dict[str, Any]
) -> dict[str, Any]:
    """Resolve a well's suggested Fimbul parameters from its metadata.

    Runs simulation.jl's well_to_simulation_params unmodified, in the session's
    own persistent Julia kernel — the same lookup geothermal-viz's own sidebar
    used to get from its now-removed ``/api/simulation/setup`` server route.
    """
    result_path = session.output_dir / "artifacts" / f"sim-setup-{uuid.uuid4().hex[:8]}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    code = _render_template(
        _SETUP_PARAMS_TEMPLATE,
        JL_PATH=Path(simulation_jl_path).resolve().as_posix(),
        PROPS_JSON=json.dumps(properties),
        RESULT_PATH=result_path.as_posix(),
    )
    result = await session.julia.eval(code)
    if result.error:
        raise RuntimeError(result.error)
    return json.loads(result_path.read_text(encoding="utf-8"))


# Holds a reference to each fire-and-forget warmup task so asyncio doesn't GC
# it mid-flight (it owns no other reference once start_simulation_warmup returns).
_warmup_tasks: set[asyncio.Task[None]] = set()

_WARMUP_TEMPLATE = """
try
    if !isdefined(Main, :well_to_simulation_params)
        include(raw"__JL_PATH__")
    end
catch
end
"""

# Loaded lazily after setup_simulation responds (see _start_cairo_warmup) so
# CairoMakie's slow first-load doesn't block "Setup Simulation".
_CAIRO_WARMUP_CODE = """
try
    using CairoMakie
catch
end
"""

# Per-session guard: CairoMakie only needs to be kicked off once.
_cairo_warmup_started: set[str] = set()


async def _warm_simulation_jl(julia: Any, simulation_jl_path: str, *, attempts: int = 30) -> None:
    """Include simulation.jl into the kernel, retrying until the Fimbul warmup
    releases it. CairoMakie is intentionally excluded here — it is deferred to
    _start_cairo_warmup so it doesn't block setup_simulation.
    """
    code = _render_template(_WARMUP_TEMPLATE, JL_PATH=Path(simulation_jl_path).resolve().as_posix())
    for attempt in range(attempts):
        try:
            await julia.eval(code)
            return
        except Exception:
            if attempt == attempts - 1:
                raise
            await asyncio.sleep(1.0)


def start_simulation_warmup(session: Session, simulation_jl_path: str) -> None:
    """Fire-and-forget: include simulation.jl at session start so setup_simulation
    finds it already loaded. CairoMakie is deferred — see _start_cairo_warmup.
    """

    async def _run() -> None:
        with contextlib.suppress(Exception):
            await _warm_simulation_jl(session.julia, simulation_jl_path)

    task = asyncio.create_task(_run())
    _warmup_tasks.add(task)
    task.add_done_callback(_warmup_tasks.discard)


def _start_cairo_warmup(session: Session) -> None:
    """Fire-and-forget: load CairoMakie in the background after setup_simulation
    has already responded, so it's ready by the time view_simulation_result runs.
    Idempotent — safe to call from every setup_simulation response.
    """
    if session.session_id in _cairo_warmup_started:
        return
    _cairo_warmup_started.add(session.session_id)

    async def _run() -> None:
        with contextlib.suppress(Exception):
            await session.julia.eval(_CAIRO_WARMUP_CODE)

    task = asyncio.create_task(_run())
    _warmup_tasks.add(task)
    task.add_done_callback(_warmup_tasks.discard)


def _summarize_simulation_result(
    case_type: str, parameters: dict[str, Any], result: dict[str, Any]
) -> str:
    wells = ", ".join(result.get("well_data", {}).keys()) or "none"
    return (
        f"{case_type} simulation completed: {result.get('num_steps', '?')} timesteps, "
        f"well(s): {wells}. Parameters used: {parameters}."
    )


# Placeholder-substituted like the Julia templates above (not an f-string) so the
# literal JS/CSS braces below don't need escaping. The well-output chart and the
# reservoir-state playback are ported straight from geothermal-viz's old sidebar
# Results tab (git history, web/js/simulation.js pre-removal): same dropdowns,
# same canvas line-chart drawing, same step/delta controls — except all reading
# from the JSON embedded below instead of fetching a live API, since that API
# (geothermal-viz's own simulation server) no longer runs simulations.
_REPORT_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  margin: 2rem; color: #1f2328;
}
table { border-collapse: collapse; margin: 1rem 0; }
td { padding: 4px 14px; border-bottom: 1px solid #e3e3df; }
h1 { font-size: 1.4rem; }
h2 {
  font-size: 1.1rem; margin-top: 2rem;
  border-top: 1px solid #e3e3df; padding-top: 1rem;
}
.controls {
  display: flex; gap: 1.5rem; align-items: center;
  margin: 0.5rem 0 1rem; flex-wrap: wrap;
}
.controls label { font-size: 0.85rem; color: #475569; margin-right: 0.35rem; }
select { padding: 4px 8px; border: 1px solid #cbd5e1; border-radius: 4px; }
.chart-wrap {
  border: 1px solid #e2e8f0; border-radius: 6px;
  padding: 8px; max-width: 900px;
}
.playback { display: flex; align-items: center; gap: 0.5rem; margin: 0.5rem 0; }
.playback button {
  border: 1px solid #cbd5e1; background: #f8fafc;
  border-radius: 4px; padding: 4px 10px; cursor: pointer;
}
.playback button:hover { background: #eef2f7; }
.playback input[type=range] { flex: 1; max-width: 300px; }
.step-label { font-size: 0.85rem; color: #475569; white-space: nowrap; }
.delta-label { font-size: 0.85rem; color: #475569; }
.reservoir-image-wrap { max-width: 700px; }
.reservoir-image-wrap img { max-width: 100%; border: 1px solid #e2e8f0; border-radius: 6px; }
.muted { color: #94a3b8; font-size: 0.85rem; }
</style></head><body>
<h1>__TITLE__</h1>
<p>__MESSAGE__</p>
<p><strong>Wells:</strong> __WELLS__ &middot; <strong>Timesteps:</strong> __NUM_STEPS__</p>
<table>__ROWS__</table>

<h2>Well Output</h2>
<div class="controls">
  <span><label for="well-select">Well</label><select id="well-select"></select></span>
  <span><label for="var-select">Variable</label><select id="var-select"></select></span>
</div>
<div class="chart-wrap"><canvas id="chart-canvas"></canvas></div>

<div id="reservoir-section" style="display:none;">
<h2>Reservoir States</h2>
<div class="controls">
  <span>
    <label for="reservoir-var-select">Variable</label>
    <select id="reservoir-var-select"></select>
  </span>
  <span class="delta-label">
    <label><input type="checkbox" id="show-delta"> Show difference from initial state</label>
  </span>
</div>
<div class="playback">
  <button id="step-first" title="First step">|&lt;</button>
  <button id="step-prev" title="Previous step">&lt;</button>
  <button id="step-next" title="Next step">&gt;</button>
  <button id="step-last" title="Last step">&gt;|</button>
  <input type="range" id="step-slider" min="0" max="0" value="0">
  <span class="step-label" id="step-label"></span>
</div>
<p class="muted">Only a sampled subset of steps was pre-rendered with the result
  (rendering every step would make this report very large).</p>
<div class="reservoir-image-wrap">
  <img id="reservoir-image" alt="Reservoir state visualization" style="display:none;">
  <p id="reservoir-placeholder" class="muted">No image for this step/variable/delta combination.</p>
</div>
</div>

<script id="sim-result-data" type="application/json">__DATA_JSON__</script>
<script>
const result = JSON.parse(document.getElementById("sim-result-data").textContent);

function populateVarSelect(wellName) {
  const select = document.getElementById("var-select");
  select.innerHTML = "";
  const wellVars = (result.well_data && result.well_data[wellName]) || {};
  for (const vname of Object.keys(wellVars)) {
    const opt = document.createElement("option");
    opt.value = vname;
    opt.textContent = vname;
    select.appendChild(opt);
  }
}

function drawResultChart() {
  const wellName = document.getElementById("well-select").value;
  const varName = document.getElementById("var-select").value;
  if (!wellName || !varName) return;
  const wellVars = result.well_data[wellName];
  if (!wellVars || !wellVars[varName]) return;

  const values = wellVars[varName];
  const timestamps = result.timestamps;
  const canvas = document.getElementById("chart-canvas");
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.parentElement.getBoundingClientRect();
  const w = Math.max(200, rect.width - 10);
  const h = 320;

  canvas.style.width = w + "px";
  canvas.style.height = h + "px";
  canvas.width = w * dpr;
  canvas.height = h * dpr;

  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const DAYS_PER_YEAR = 365.25;
  const timeVals = timestamps.map(t => t / DAYS_PER_YEAR);
  const tMin = Math.min(...timeVals);
  const tMax = Math.max(...timeVals);
  let vMin = Math.min(...values);
  let vMax = Math.max(...values);
  if (vMin === vMax) { vMin -= 1; vMax += 1; }
  const vPad = (vMax - vMin) * 0.05;
  vMin -= vPad;
  vMax += vPad;

  const pad = { top: 15, right: 15, bottom: 42, left: 70 };
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;

  ctx.fillStyle = "#fafbfc";
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = "#94a3b8";
  ctx.lineWidth = 1;
  ctx.strokeRect(pad.left, pad.top, plotW, plotH);

  ctx.strokeStyle = "#e2e8f0";
  ctx.lineWidth = 0.5;
  for (let i = 1; i < 5; i++) {
    const y = pad.top + (plotH * i / 5);
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + plotW, y); ctx.stroke();
  }
  for (let i = 1; i < 5; i++) {
    const x = pad.left + (plotW * i / 5);
    ctx.beginPath(); ctx.moveTo(x, pad.top); ctx.lineTo(x, pad.top + plotH); ctx.stroke();
  }

  const tRange = (tMax - tMin) || 1;
  const vRange = (vMax - vMin) || 1;
  ctx.strokeStyle = "#2563eb";
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < values.length; i++) {
    const x = pad.left + ((timeVals[i] - tMin) / tRange) * plotW;
    const y = pad.top + (1 - (values[i] - vMin) / vRange) * plotH;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();

  ctx.save();
  ctx.fillStyle = "#475569";
  ctx.font = "11px -apple-system, BlinkMacSystemFont, sans-serif";
  ctx.textAlign = "center";
  ctx.translate(13, pad.top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(varName, 0, 0);
  ctx.restore();

  ctx.fillStyle = "#475569";
  ctx.font = "11px -apple-system, BlinkMacSystemFont, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("Time [years]", pad.left + plotW / 2, h - 3);

  ctx.fillStyle = "#64748b";
  ctx.font = "10px -apple-system, BlinkMacSystemFont, sans-serif";
  ctx.textAlign = "right";
  for (let i = 0; i <= 5; i++) {
    const val = vMax - (i / 5) * (vMax - vMin);
    const y = pad.top + (plotH * i / 5);
    ctx.fillText(val.toFixed(1), pad.left - 6, y + 4);
  }
  ctx.textAlign = "center";
  for (let i = 0; i <= 5; i++) {
    const val = tMin + (i / 5) * (tMax - tMin);
    const x = pad.left + (plotW * i / 5);
    ctx.fillText(val.toFixed(1), x, pad.top + plotH + 16);
  }
}

const wellNames = Object.keys(result.well_data || {});
const wellSelect = document.getElementById("well-select");
for (const wname of wellNames) {
  const opt = document.createElement("option");
  opt.value = wname; opt.textContent = wname;
  wellSelect.appendChild(opt);
}
if (wellNames.length) {
  populateVarSelect(wellNames[0]);
  wellSelect.addEventListener("change", () => {
    populateVarSelect(wellSelect.value);
    drawResultChart();
  });
  document.getElementById("var-select").addEventListener("change", drawResultChart);
  window.addEventListener("resize", drawResultChart);
  drawResultChart();
}

const reservoirImages = result.reservoir_images || {};
const reservoirSteps = (result.reservoir_steps || []).slice().sort((a, b) => a - b);
const reservoirVars = Object.keys(reservoirImages)
  .filter(v => Object.keys(reservoirImages[v] || {}).length > 0);

if (reservoirVars.length && reservoirSteps.length) {
  document.getElementById("reservoir-section").style.display = "block";
  const varSelect = document.getElementById("reservoir-var-select");
  for (const v of reservoirVars) {
    const opt = document.createElement("option");
    opt.value = v; opt.textContent = v;
    varSelect.appendChild(opt);
  }
  const slider = document.getElementById("step-slider");
  slider.min = 0;
  slider.max = reservoirSteps.length - 1;
  slider.value = reservoirSteps.length - 1;

  function updateReservoirImage() {
    const varName = varSelect.value;
    const delta = document.getElementById("show-delta").checked;
    const step = reservoirSteps[parseInt(slider.value, 10)];
    const img = document.getElementById("reservoir-image");
    const placeholder = document.getElementById("reservoir-placeholder");
    const byStep = (reservoirImages[varName] || {})[String(step)] || {};
    const b64 = byStep[delta ? "true" : "false"];
    document.getElementById("step-label").textContent = "Step " + step + " / " + result.num_steps;
    if (b64) {
      img.src = "data:image/png;base64," + b64;
      img.style.display = "block";
      placeholder.style.display = "none";
    } else {
      img.style.display = "none";
      placeholder.style.display = "block";
    }
  }

  function setStepIndex(idx) {
    idx = Math.max(0, Math.min(idx, reservoirSteps.length - 1));
    slider.value = idx;
    updateReservoirImage();
  }

  varSelect.addEventListener("change", updateReservoirImage);
  document.getElementById("show-delta").addEventListener("change", updateReservoirImage);
  slider.addEventListener("input", updateReservoirImage);
  document.getElementById("step-first").addEventListener("click", () => setStepIndex(0));
  document.getElementById("step-prev").addEventListener("click", () => {
    setStepIndex(parseInt(slider.value, 10) - 1);
  });
  document.getElementById("step-next").addEventListener("click", () => {
    setStepIndex(parseInt(slider.value, 10) + 1);
  });
  document.getElementById("step-last").addEventListener("click", () => {
    setStepIndex(reservoirSteps.length - 1);
  });
  updateReservoirImage();
}
</script>
</body></html>"""


def _build_simulation_report_html(
    case_type: str, parameters: dict[str, Any], result: dict[str, Any]
) -> str:
    import html

    rows = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>"
        for k, v in parameters.items()
    )
    wells = ", ".join(result.get("well_data", {}).keys()) or "none"
    title = html.escape(case_type) + " simulation results"
    # </script> inside the embedded JSON (won't normally occur, but a stray well
    # or parameter name could in principle contain it) would otherwise close the
    # tag early.
    data_json = json.dumps(result).replace("</", "<\\/")
    return _render_template(
        _REPORT_TEMPLATE,
        TITLE=title,
        MESSAGE=html.escape(str(result.get("message", ""))),
        WELLS=html.escape(wells),
        NUM_STEPS=str(result.get("num_steps", "?")),
        ROWS=rows,
        DATA_JSON=data_json,
    )


def _record_simulation_artifact(
    session: Session, case_type: str, parameters: dict[str, Any], result: dict[str, Any]
) -> str:
    """Pin the results as a new report tab in the chat's canvas (not the map's
    own sidebar). Each run gets its own file, slot, and caption number, so a
    later run opens an additional tab instead of overwriting the previous
    run's results out from under it."""
    run_no = _simulation_run_count.get(session.session_id, 0) + 1
    _simulation_run_count[session.session_id] = run_no
    rel = f"artifacts/simulation-results-{run_no}.html"
    out = session.output_dir / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_build_simulation_report_html(case_type, parameters, result), encoding="utf-8")
    session.trace.append(
        "artifact",
        {
            "path": rel,
            "mime": "text/html",
            "format": "html",
            "caption": f"{case_type} simulation results #{run_no}",
            "kind": "report",
            "slot": f"simulation-results-{run_no}",
        },
    )
    return rel


def _make_run_simulation_tool(session: Session, simulation_jl_path: str):
    # Reuses the same ContextVar-based streaming writer run_julia uses, so this
    # tool's progress shows up live in its tool card exactly the same way.
    from jutul_agent.agent.tools import _capture_delta_writer

    @tool
    async def run_simulation(case_type: str, parameters: dict[str, float]) -> str:
        """Run a Fimbul geothermal simulation (an AGS or BTES case).

        Args:
            case_type: "AGS" (single energy well) or "BTES" (well-park array).
            parameters: Simulation parameters by name (e.g. well_depth,
                surface_temperature, geothermal_gradient, flow_rate, num_years,
                ...). Prefer values already resolved for the well in question
                (e.g. from a well's metadata) over invented ones, and confirm
                with the user before changing key parameters yourself.
        """
        try:
            result = await _execute_fimbul_simulation(
                session, simulation_jl_path, case_type, parameters, on_chunk=_capture_delta_writer()
            )
        except Exception as exc:
            return f"ERROR: simulation failed: {exc}"
        if result.get("status") != "completed":
            return f"ERROR: {result.get('message', 'simulation failed')}"
        _record_simulation_artifact(session, case_type, parameters, result)
        return _summarize_simulation_result(case_type, parameters, result)

    return run_simulation


def _make_view_simulation_result_tool(session: Session, simulation_jl_path: str):
    @tool
    async def view_simulation_result(
        var: str = "Temperature", step: int = -1, delta: bool = False
    ) -> str | list[dict[str, Any]]:
        """Show a reservoir-state image from the most recent simulation.

        Lets you actually see (and describe) the reservoir field, rather than
        only knowing the summary numbers — call this when asked what a result
        "looks like". Errors if no simulation has run yet this session.

        Args:
            var: Reservoir variable to render (e.g. "Temperature", "Pressure").
            step: 1-based timestep to show; -1 (default) for the last step.
            delta: Show the change from the initial state instead of the
                absolute value.
        """
        try:
            b64 = await _render_simulation_view(session, simulation_jl_path, var, step, delta)
        except Exception as exc:
            return f"ERROR: {exc}"
        if not b64:
            return "No simulation result is available yet — run a simulation first."
        return [
            {"type": "text", "text": f"Reservoir {var} at the requested step."},
            {"type": "image", "mime_type": "image/png", "base64": b64},
        ]

    return view_simulation_result


def make_run_simulation_action(simulation_jl_path: str):
    """The direct (non-LLM) counterpart to ``run_simulation``.

    For when a front end (the map panel's sidebar) already has exact,
    structured parameters chosen by the user — nothing for the model to
    interpret, so this bypasses it entirely (see
    jutul_agent.interfaces.server.app.ActionHandler) rather than risking the
    model paraphrasing or re-deriving the parameters from a synthetic prompt.
    It still sends the same `tool`-shaped wire messages a real tool call would
    (started/delta/finished), so it looks identical in the chat.
    """

    async def run_simulation_action(
        session: Session,
        args: dict[str, Any],
        send_wire: Callable[[dict[str, Any]], Awaitable[None]],
        queue_ui_event: Callable[[Any], None],
    ) -> None:
        case_type = str(args.get("case_type") or "")
        raw_parameters = args.get("parameters")
        parameters = raw_parameters if isinstance(raw_parameters, dict) else {}
        tool_call_id = f"sim-{uuid.uuid4().hex[:8]}"
        label = f"Run {case_type or 'Fimbul'} simulation"

        await send_wire(
            {
                "type": "tool",
                "event": "started",
                "name": "run_simulation",
                "label": label,
                "tool_call_id": tool_call_id,
                "args": {"case_type": case_type, "parameters": parameters},
                "content": None,
            }
        )

        # on_chunk fires synchronously from the kernel's own read loop; queue the
        # chunks and drain them through a consumer task so they reach send_wire
        # (a coroutine) one at a time, in order, while the simulation keeps running.
        chunk_queue: asyncio.Queue[str | None] = asyncio.Queue()

        def on_chunk(chunk: OutputChunk) -> None:
            if chunk.text:
                chunk_queue.put_nowait(chunk.text)

        async def drain_chunks() -> None:
            while True:
                item = await chunk_queue.get()
                if item is None:
                    return
                await send_wire(
                    {
                        "type": "tool",
                        "event": "delta",
                        "name": "run_simulation",
                        "tool_call_id": tool_call_id,
                        "content": item,
                    }
                )

        consumer = asyncio.create_task(drain_chunks())
        try:
            result = await _execute_fimbul_simulation(
                session, simulation_jl_path, case_type, parameters, on_chunk=on_chunk
            )
        except Exception as exc:
            await chunk_queue.put(None)
            await consumer
            await send_wire(
                {
                    "type": "tool",
                    "event": "error",
                    "name": "run_simulation",
                    "tool_call_id": tool_call_id,
                    "content": str(exc),
                }
            )
            return
        await chunk_queue.put(None)
        await consumer

        if result.get("status") != "completed":
            message = str(result.get("message") or "simulation failed")
            await send_wire(
                {
                    "type": "tool",
                    "event": "error",
                    "name": "run_simulation",
                    "tool_call_id": tool_call_id,
                    "content": message,
                }
            )
            return

        _record_simulation_artifact(session, case_type, parameters, result)
        # The new report tab: the same mechanism a real tool's artifact gets,
        # since nothing else flushes side outputs outside of a real turn.
        from jutul_agent.interfaces.server.app import artifact_wire_events

        events = session.trace.iter_events()
        latest_artifact = next((e for e in reversed(events) if e.kind == "artifact"), None)
        if latest_artifact is not None:
            for wire in artifact_wire_events([latest_artifact.payload], session.session_id):
                await send_wire(wire)

        summary = _summarize_simulation_result(case_type, parameters, result)
        await send_wire(
            {
                "type": "tool",
                "event": "finished",
                "name": "run_simulation",
                "tool_call_id": tool_call_id,
                "content": summary,
            }
        )
        # The model wasn't involved in running this, so it has no memory of it —
        # fold a note into whatever the user sends next (see app.py's
        # _with_pending_ui_events), the same mechanism well clicks already use.
        queue_ui_event(
            {
                "event": "simulationCompleted",
                "case_type": case_type,
                "parameters": parameters,
                "summary": summary,
            }
        )

    return run_simulation_action


def make_setup_simulation_action(simulation_jl_path: str):
    """The direct (non-LLM) counterpart to the map's "Setup Simulation" button.

    Resolves a well's suggested Fimbul parameters from its metadata (a lookup,
    not a judgment call — nothing for the model to decide), on the session's
    own kernel, and replies with a targeted `ui` action for the map panel to
    render as a pre-filled form — the in-process replacement for today's
    dedicated-kernel ``/api/simulation/setup`` REST route.
    """

    async def setup_simulation_action(
        session: Session,
        args: dict[str, Any],
        send_wire: Callable[[dict[str, Any]], Awaitable[None]],
        queue_ui_event: Callable[[Any], None],
    ) -> None:
        try:
            result = await _setup_simulation_params(session, simulation_jl_path, args)
        except Exception as exc:
            await send_wire(
                {
                    "type": "ui",
                    "action": "simulation_setup_error",
                    "payload": {"message": str(exc)},
                    "target": _MAP_TARGET,
                }
            )
            return
        await send_wire(
            {
                "type": "ui",
                "action": "simulation_params",
                "payload": result,
                "target": _MAP_TARGET,
            }
        )
        _start_cairo_warmup(session)

    return setup_simulation_action


def _make_run_borehole_simulation_tool(session: Session):
    @tool
    async def run_borehole_simulation(  # noqa: PLR0913
        N_1: int = 1,
        N_2: int = 1,
        B: float = 5.0,
        H: float = 200.0,
        D: float = 0.5,
        r_b: float = 0.065,
        tilt: float = 0.0,
        orientation: float = 0.0,
        k_s: float = 3.7,
        alpha: float = 1.59e-6,
        r_in: float = 0.015,
        r_out: float = 0.018,
        D_s: float = 0.030,
        k_p: float = 0.38,
        k_g: float = 2.3,
        epsilon: float = 1e-4,
        COP: float = 3.5,
        G: float = 0.03,
        pattern: str = "rectangular",
        num_sectors: int | None = None,
        overrides: dict[str, dict[str, dict]] | None = None,
    ) -> str:
        """Run a pygfunction borehole heat exchanger simulation using the heating
        demands most recently generated for the current building.

        Uses the demand series from the last 'Generate energy demands' button click.
        Computes hourly carrier-fluid temperatures (T_in, T_out) for a U-tube
        borehole field. Ground temperature is derived from ERA5-Land 1991-2020
        climate normals at the building's coordinates.

        Args:
            N_1: Number of boreholes in the x direction for pattern = "rectangular";
            else, it is the total number of boreholes. Default 1.
            N_2: Number of boreholes in the y direction for pattern = "rectangular".
            If pattern = "polygonal", then it is the number of sides in the regular polygon.
            Else, it does not do anything. Default 1.
            B: Borehole spacing [m] (centre-to-centre). For the sunflower pattern this
            is the approximate nearest-neighbour spacing. Default 5.
            H: Active borehole length [m]. Default 200.
            D: Buried depth from surface to top of active section [m]. Default 0.5 (matches Fimbul).
            r_b: Borehole radius [m]. Default 0.065 (matches Fimbul).
            tilt: Borehole tilt angle from vertical [radians]. Default 0.
            orientation: Azimuthal direction of tilt [radians]. Default 0.
            k_s: Ground thermal conductivity [W/(m·K)]. Default 3.7 (matches Fimbul).
            alpha: Ground thermal diffusivity [m²/s]. Default 1.59e-6 (matches Fimbul: 3.7/(2580×900)).
            r_in: Pipe inner radius [m]. Default 0.015.
            r_out: Pipe outer radius [m]. Default 0.018 (matches Fimbul).
            D_s: Shank spacing, borehole centre to pipe centre [m]. Default 0.030 (matches Fimbul).
            k_p: Pipe wall thermal conductivity [W/(m·K)]. Default 0.38 (matches Fimbul).
            k_g: Grout thermal conductivity [W/(m·K)]. Default 2.3 (matches Fimbul).
            epsilon: Pipe inner-wall roughness [m]. Default 1e-4 (matches Fimbul/JutulDarcy).
            COP: Heat-pump coefficient of performance. Default 3.5.
            G: Geothermal gradient [K/m]. Default 0.03 (matches Fimbul).
            pattern: the pattern type for the borehole field.
            Possible values: "sunflower", "rectangular", "circular", "polygonal".
            num_sectors: Number of hydraulic sectors. Boreholes within a sector are
            connected in series; sectors are connected in parallel. Defaults to the
            total number of boreholes (one borehole per sector = fully parallel field).
            overrides: Optional per-borehole parameter changes applied after the
            field is generated from the pattern. Structure:
            {"sector_index": {"well_index": {param: value, ...}, ...}, ...}
            where sector_index and well_index are 0-based string integers. Within
            each borehole dict the following keys are recognised:
              H (float): active borehole depth [m].
              D (float): buried depth to top of active section [m].
              r_b (float): borehole radius [m].
              tilt (float): tilt from vertical [radians].
              orientation (float): azimuthal direction of tilt [radians].
              dx (float): displacement added to the pattern x-coordinate [m].
              dy (float): displacement added to the pattern y-coordinate [m].
            dx/dy shift a borehole relative to its pattern-generated position;
            all other keys replace the uniform value directly.
            Example — deepen well 1 in sector 0 and shift well 2 by (2, 3) m:
            {"0": {"1": {"H": 250.0}, "2": {"dx": 2.0, "dy": 3.0}}}
        """
        import pandas as pd

        from backend_building_selection import get_session_demand_data
        from pygfunction_sim import (
            GroundParams,
            PipeParams,
            rectangle_field,
            sunflower_field,
            circular_field, 
            polygonal_field,
            simulate_borehole_temperatures,
        )

        demand = get_session_demand_data(session.session_id)
        if demand is None:
            return (
                "No energy demand data is available for this session. "
                "Ask the user to click 'Generate energy demands' for a building "
                "on the map first."
            )

        df = pd.DataFrame({"time": demand["timestamps"], "kW": demand["demand_kw"]})
        try:
            if pattern == "rectangular":
                boreholes = rectangle_field(N_1, N_2, B, H, D, r_b, tilt, orientation, num_sectors)
            elif pattern == "sunflower":
                boreholes = sunflower_field(N_1, B, H, D, r_b, tilt, orientation, num_sectors)
            elif pattern == "circular":
                boreholes = circular_field(N_1, B, H, D, r_b, tilt, orientation, num_sectors)
            elif pattern == "polygonal":
                boreholes = polygonal_field(N_1, B, N_2, H, D, r_b, tilt, orientation, num_sectors)
            else:
                raise ValueError(
                    f"Unknown pattern {pattern!r}. Valid options: 'rectangular', 'sunflower', 'circular', 'polygonal'."
                )
            boreholes = _apply_overrides(boreholes, overrides)

            df_result, gfunc_time, gfunc_vals = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: simulate_borehole_temperatures(
                    df,
                    lat=demand.get("lat"),
                    lon=demand.get("lon"),
                    boreholes=boreholes,
                    ground=GroundParams(k_s=k_s, alpha=alpha),
                    pipe=PipeParams(r_in=r_in, r_out=r_out, D_s=D_s, k_p=k_p, k_g=k_g, epsilon=epsilon),
                    COP=COP,
                    G=G,
                ),
            )
        except Exception as exc:
            return f"ERROR: Borehole simulation failed: {exc}"

        T_in = df_result["T_in"].tolist()
        T_out = df_result["T_out"].tolist()
        timestamps = df_result["time"].astype(str).tolist()
        n = len(T_in)

        run_no = _borehole_sim_run_count.get(session.session_id, 0) + 1
        _borehole_sim_run_count[session.session_id] = run_no
        _session_borehole_results[session.session_id] = {
            "run_no": run_no,
            "timestamps": timestamps,
            "T_in": T_in,
            "T_out": T_out,
            "gfunc_time": gfunc_time.tolist(),
            "gfunc_vals": gfunc_vals.tolist(),
        }

        T_in_daily, _ = _aggregate_tin_profile(T_in, window_hours=24)
        below_zero_days = [i + 1 for i, t in enumerate(T_in_daily) if t < 0.0]
        if below_zero_days:
            if len(below_zero_days) <= 30:
                below_zero_str = f"days {', '.join(map(str, below_zero_days))}"
            else:
                below_zero_str = (
                    f"days {below_zero_days[0]}–{below_zero_days[-1]} "
                    f"({len(below_zero_days)} days total)"
                )
            below_zero_note = f"WARNING: {len(below_zero_days)} daily-avg T_in period(s) below 0 °C: {below_zero_str}\n"
        else:
            below_zero_note = "All daily-avg T_in periods are above 0 °C.\n"

        return (
            f"Borehole simulation completed: {n} hourly timesteps, "
            f"{len(boreholes)} borehole(s), depth {H} m.\n"
            f"T_in:  min {min(T_in):.1f} °C  max {max(T_in):.1f} °C  "
            f"mean {sum(T_in)/n:.1f} °C\n"
            f"T_out: min {min(T_out):.1f} °C  max {max(T_out):.1f} °C  "
            f"mean {sum(T_out)/n:.1f} °C\n"
            + below_zero_note +
            "Use plot_borehole_temperatures or plot_borehole_gfunction if the "
            "user wants to visualise the results."
        )

    return run_borehole_simulation


def _make_sweep_borehole_parameters_tool(session: Session):  # noqa: PLR0912
    @tool
    async def sweep_borehole_parameters(  # noqa: PLR0913
        N_1: list[int] = [1],
        N_2: list[int] = [1],
        B: list[float] = [5.0],
        H: list[float] = [200.0],
        D: list[float] = [0.5],
        r_b: list[float] = [0.065],
        tilt: list[float] = [0.0],
        orientation: list[float] = [0.0],
        k_s: list[float] = [3.7],
        alpha: list[float] = [1.59e-6],
        r_in: list[float] = [0.015],
        r_out: list[float] = [0.018],
        D_s: list[float] = [0.030],
        k_p: list[float] = [0.38],
        k_g: list[float] = [2.3],
        epsilon: list[float] = [1e-4],
        COP: list[float] = [3.5],
        G: list[float] = [0.03],
        pattern: str = "rectangular",
        num_sectors: int | None = None,
        overrides: dict[str, dict[str, dict[str, list]]] | None = None,
    ) -> str:
        """Run pygfunction borehole simulations for every combination of the given
        parameter lists and report summary temperatures (min/mean/max T_in and T_out)
        for each combination.

        Only call this when the user explicitly asks for a parameter sweep or wants
        to compare multiple borehole configurations at once. For a single
        configuration, use run_borehole_simulation instead.

        Each parameter accepts a list of values; one simulation is run per element
        of the Cartesian product of all lists. Parameters with a single-element list
        are held fixed across all combinations. The total number of simulations equals
        the product of all list lengths — keep individual lists short to limit runtime.

        Args:
            N_1: List of values for number of boreholes in x direction (rectangular)
                or total number of boreholes (other patterns). Default [1].
            N_2: List of values for number of boreholes in y direction (rectangular)
                or number of polygon sides (polygonal). Default [1].
            B: List of values for borehole spacing [m] or sunflower radius [m]. Default [5.0].
            H: List of values for active borehole length [m]. Default [200.0].
            D: List of values for buried depth [m]. Default [0.5] (matches Fimbul).
            r_b: List of values for borehole radius [m]. Default [0.065] (matches Fimbul).
            tilt: List of values for tilt angle from vertical [radians]. Default [0.0].
            orientation: List of values for azimuthal direction of tilt [radians]. Default [0.0].
            k_s: List of values for ground thermal conductivity [W/(m·K)]. Default [3.7] (matches Fimbul).
            alpha: List of values for ground thermal diffusivity [m²/s]. Default [1.59e-6] (matches Fimbul).
            r_in: List of values for pipe inner radius [m]. Default [0.015].
            r_out: List of values for pipe outer radius [m]. Default [0.018] (matches Fimbul).
            D_s: List of values for shank spacing [m]. Default [0.030] (matches Fimbul).
            k_p: List of values for pipe wall thermal conductivity [W/(m·K)]. Default [0.38] (matches Fimbul).
            k_g: List of values for grout thermal conductivity [W/(m·K)]. Default [2.3] (matches Fimbul).
            epsilon: List of values for pipe roughness [m]. Default [1e-4] (matches Fimbul/JutulDarcy).
            COP: List of values for heat-pump COP. Default [3.5].
            G: List of values for geothermal gradient [K/m]. Default [0.03] (matches Fimbul).
            pattern: Borehole field pattern shared across all combinations.
                One of: "rectangular", "sunflower", "circular", "polygonal".
            num_sectors: Number of hydraulic sectors (fixed across all combinations).
                Boreholes within a sector are connected in series; sectors in parallel.
                Defaults to total number of boreholes (fully parallel field).
            overrides: Optional per-borehole parameter changes included in the sweep.
                Same nested structure as run_borehole_simulation's overrides, but each
                parameter value is a list that participates in the Cartesian product:
                {"sector_index": {"well_index": {"param": [v1, v2, ...], ...}, ...}, ...}
                Recognised per-borehole keys (each a list of floats):
                  H: active borehole depth [m].
                  D: buried depth [m].
                  r_b: borehole radius [m].
                  tilt: tilt from vertical [radians].
                  orientation: azimuthal direction of tilt [radians].
                  dx: list of displacements added to the pattern x-coordinate [m].
                  dy: list of displacements added to the pattern y-coordinate [m].
                Each (sector, well, param) triple is an independent sweep dimension.
                Example — sweep over two depths for well 1 in sector 0, and two displacements in the 
                x- and y-directions for well 2 in sector 0:
                {"0": {"1": {"H": [200.0,250.0]}, "2": {"dx": [1.0,2.0], "dy": [3.0,5.0]}}}

        """
        import itertools

        import pandas as pd

        from backend_building_selection import get_session_demand_data
        from pygfunction_sim import (
            GroundParams,
            PipeParams,
            circular_field,
            polygonal_field,
            rectangle_field,
            simulate_borehole_temperatures,
            sunflower_field,
        )

        demand = get_session_demand_data(session.session_id)
        if demand is None:
            return (
                "No energy demand data is available for this session. "
                "Ask the user to click 'Generate energy demands' for a building "
                "on the map first."
            )

        if pattern not in ("rectangular", "sunflower", "circular", "polygonal"):
            return (
                f"Unknown pattern {pattern!r}. "
                "Valid options: 'rectangular', 'sunflower', 'circular', 'polygonal'."
            )

        df = pd.DataFrame({"time": demand["timestamps"], "kW": demand["demand_kw"]})

        param_names = [
            "N_1", "N_2", "B", "H", "D", "r_b", "tilt", "orientation",
            "k_s", "alpha", "r_in", "r_out", "D_s", "k_p", "k_g", "epsilon", "COP", "G",
        ]
        param_lists = [
            N_1, N_2, B, H, D, r_b, tilt, orientation,
            k_s, alpha, r_in, r_out, D_s, k_p, k_g, epsilon, COP, G,
        ]
        n_base = len(param_names)

        # Flatten overrides into additional sweep dimensions.
        # Each (sector, well, param) triple becomes one entry in param_names/param_lists.
        # The label "[s,w].param" appears in the varied/fixed output.
        override_dims: list[tuple[str, str, str]] = []  # (sector_str, well_str, param)
        if overrides:
            for s_str, wells in overrides.items():
                for w_str, changes in wells.items():
                    for param_key, values in changes.items():
                        vals = values if isinstance(values, list) else [values]
                        override_dims.append((s_str, w_str, param_key))
                        param_names.append(f"[{s_str},{w_str}].{param_key}")
                        param_lists.append([float(v) for v in vals])

        varied = [name for name, vals in zip(param_names, param_lists) if len(vals) > 1]
        fixed = {
            name: vals[0]
            for name, vals in zip(param_names, param_lists)
            if len(vals) == 1
        }
        combos = list(itertools.product(*param_lists))
        n_combos = len(combos)

        rows: list[dict] = []
        for combo in combos:
            base_combo = combo[:n_base]
            ov_vals = combo[n_base:]
            (c_N_1, c_N_2, c_B, c_H, c_D, c_r_b, c_tilt, c_orientation,
             c_k_s, c_alpha, c_r_in, c_r_out, c_D_s, c_k_p, c_k_g, c_epsilon,
             c_COP, c_G) = base_combo
            row: dict = dict(zip(param_names, combo))

            # Reconstruct a scalar overrides dict for this combo.
            combo_overrides: dict[str, dict[str, dict]] = {}
            for (s_str, w_str, param_key), val in zip(override_dims, ov_vals):
                combo_overrides.setdefault(s_str, {}).setdefault(w_str, {})[param_key] = val

            try:
                if pattern == "rectangular":
                    boreholes = rectangle_field(c_N_1, c_N_2, c_B, c_H, c_D, c_r_b, c_tilt, c_orientation, num_sectors)
                elif pattern == "sunflower":
                    boreholes = sunflower_field(c_N_1, c_B, c_H, c_D, c_r_b, c_tilt, c_orientation, num_sectors)
                elif pattern == "circular":
                    boreholes = circular_field(c_N_1, c_B, c_H, c_D, c_r_b, c_tilt, c_orientation, num_sectors)
                elif pattern == "polygonal":
                    boreholes = polygonal_field(c_N_1, c_B, c_N_2, c_H, c_D, c_r_b, c_tilt, c_orientation, num_sectors)
                else:
                    raise ValueError(
                        f"Unknown pattern {pattern!r}. Valid options: 'rectangular', 'sunflower', 'circular', 'polygonal'."
                    )
                boreholes = _apply_overrides(boreholes, combo_overrides or None)
                df_result, _, _ = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: simulate_borehole_temperatures(
                        df,
                        lat=demand.get("lat"),
                        lon=demand.get("lon"),
                        boreholes=boreholes,
                        ground=GroundParams(k_s=c_k_s, alpha=c_alpha),
                        pipe=PipeParams(
                            r_in=c_r_in, r_out=c_r_out, D_s=c_D_s,
                            k_p=c_k_p, k_g=c_k_g, epsilon=c_epsilon,
                        ),
                        COP=c_COP,
                        G=c_G,
                    ),
                )
                T_in = df_result["T_in"].tolist()
                T_out = df_result["T_out"].tolist()
                n = len(T_in)
                row.update(
                    n_boreholes=len(boreholes),
                    T_in_min=min(T_in),
                    T_in_mean=sum(T_in) / n,
                    T_in_max=max(T_in),
                    T_out_min=min(T_out),
                    T_out_mean=sum(T_out) / n,
                    T_out_max=max(T_out),
                    error=None,
                )
            except Exception as exc:
                row["error"] = str(exc)
            rows.append(row)

        # Build output text
        lines: list[str] = [
            f"Parameter sweep: {n_combos} combination(s), pattern={pattern!r}."
        ]
        if fixed:
            fixed_str = "  ".join(
                f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                for k, v in fixed.items()
            )
            lines.append(f"Fixed: {fixed_str}")
        if varied:
            lines.append(f"Varied: {', '.join(varied)}")
        lines.append("")

        # Table header — only varied params + n_boreholes + results
        w = 9
        header = (
            f"  {'#':>3}"
            + "".join(f"  {name:>{w}}" for name in varied)
            + f"  {'n_bh':>5}"
            + f"  {'Ti_min':>{w}}  {'Ti_mean':>{w}}  {'Ti_max':>{w}}"
            + f"  {'To_min':>{w}}  {'To_mean':>{w}}  {'To_max':>{w}}"
        )
        lines.append(header)
        lines.append("-" * len(header))

        n_errors = 0
        for i, row in enumerate(rows):
            if row.get("error"):
                n_errors += 1
                varied_vals = "  ".join(
                    f"{row[name]:>{w}.4g}" if isinstance(row[name], float) else f"{row[name]:>{w}}"
                    for name in varied
                )
                lines.append(f"  {i+1:>3}  {varied_vals}  ERROR: {row['error']}")
                continue
            varied_vals = "".join(
                f"  {row[name]:>{w}.4g}" if isinstance(row[name], float) else f"  {row[name]:>{w}}"
                for name in varied
            )
            lines.append(
                f"  {i+1:>3}"
                + varied_vals
                + f"  {row['n_boreholes']:>5}"
                + f"  {row['T_in_min']:>{w}.2f}  {row['T_in_mean']:>{w}.2f}  {row['T_in_max']:>{w}.2f}"
                + f"  {row['T_out_min']:>{w}.2f}  {row['T_out_mean']:>{w}.2f}  {row['T_out_max']:>{w}.2f}"
            )

        if n_errors:
            lines.append(f"\n{n_errors} combination(s) failed (see ERROR lines above).")
        lines.append(
            "\nTo plot results for the best configuration, call run_borehole_simulation "
            "with those parameter values."
        )
        return "\n".join(lines)

    return sweep_borehole_parameters


def _build_temperatures_html(timestamps: list[str], T_in: list[float], T_out: list[float]) -> str:
    import json

    stride = max(1, len(timestamps) // 365)
    data = json.dumps({
        "labels": timestamps[::stride],
        "T_in": T_in[::stride],
        "T_out": T_out[::stride],
    })
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Borehole fluid temperatures</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>body{{font-family:sans-serif;margin:24px;background:#f5f5f5}}
.chart-wrap{{background:#fff;border-radius:8px;padding:20px;box-shadow:0 1px 4px rgba(0,0,0,.1);margin-bottom:24px}}</style>
</head><body>
<h1 style="font-size:1.1rem;color:#333;margin-bottom:16px">Borehole fluid temperatures</h1>
<div class="chart-wrap"><canvas id="c"></canvas></div>
<script>
const MONTHS=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
function fmtDate(s){{const d=new Date(s);return isNaN(d)?s:String(d.getDate()).padStart(2,'0')+' '+MONTHS[d.getMonth()];}}
const d={data};
new Chart(document.getElementById('c'),{{
  type:'line',
  data:{{labels:d.labels,datasets:[
    {{label:'Tᵢₙ (°C)',data:d.T_in,borderColor:'#2980b9',borderWidth:1.5,pointRadius:0,tension:0.1,fill:false}},
    {{label:'Tₒᵤₜ (°C)',data:d.T_out,borderColor:'#27ae60',borderWidth:1.5,pointRadius:0,tension:0.1,fill:false}},
  ]}},
  options:{{responsive:true,animation:false,plugins:{{legend:{{position:'top'}}}},
    scales:{{
      x:{{title:{{display:true,text:'Date'}},ticks:{{maxTicksLimit:12,maxRotation:0,callback(v){{return fmtDate(this.getLabelForValue(v));}}}}  }},
      y:{{title:{{display:true,text:'Temperature (°C)'}}}}
    }}}}
}});
</script></body></html>"""


def _build_gfunction_html(time_s: list[float], gfunc: list[float]) -> str:
    import json, math

    data = json.dumps({
        "labels": [f"{math.log(t):.2f}" for t in time_s],
        "g": [round(v, 4) for v in gfunc],
    })
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>g-function</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>body{{font-family:sans-serif;padding:16px}}canvas{{max-height:420px}}</style>
</head><body>
<h2>g-function</h2>
<canvas id="c"></canvas>
<script>
const d={data};
new Chart(document.getElementById('c'),{{
  type:'line',
  data:{{labels:d.labels,datasets:[
    {{label:'g(ln t)',data:d.g,borderColor:'#2ecc71',pointRadius:3,tension:0}},
  ]}},
  options:{{animation:false,plugins:{{legend:{{position:'top'}}}},
    scales:{{x:{{title:{{display:true,text:'ln(t [s])'}}}},y:{{title:{{display:true,text:'g'}}}}}}}}
}});
</script></body></html>"""


def _make_plot_borehole_temperatures_tool(session: Session):
    @tool
    async def plot_borehole_temperatures() -> str:
        """Plot T_in and T_out vs time from the most recent borehole simulation.

        Opens a chart in a dedicated canvas tab that the user can see. Call run_borehole_simulation first.
        """
        result = _session_borehole_results.get(session.session_id)
        if result is None:
            return "No borehole simulation results available. Run run_borehole_simulation first."

        run_no = result["run_no"]
        html = _build_temperatures_html(result["timestamps"], result["T_in"], result["T_out"])
        rel = f"artifacts/borehole-temperatures-{run_no}.html"
        out = session.output_dir / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        session.trace.append(
            "artifact",
            {
                "path": rel,
                "mime": "text/html",
                "format": "html",
                "caption": f"Borehole temperatures #{run_no}",
                "kind": "report",
                "slot": f"borehole-temperatures-{run_no}",
            },
        )
        return "Temperature plot opened in canvas."

    return plot_borehole_temperatures


def _make_plot_borehole_gfunction_tool(session: Session):
    @tool
    async def plot_borehole_gfunction() -> str:
        """Plot the g-function vs ln(t) for the borehole field from the most recent simulation.

        Opens a chart in a dedicated canvas tab that the user can see. Call run_borehole_simulation first.
        """
        result = _session_borehole_results.get(session.session_id)
        if result is None:
            return "No borehole simulation results available. Run run_borehole_simulation first."

        run_no = result["run_no"]
        html = _build_gfunction_html(result["gfunc_time"], result["gfunc_vals"])
        rel = f"artifacts/borehole-gfunction-{run_no}.html"
        out = session.output_dir / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        session.trace.append(
            "artifact",
            {
                "path": rel,
                "mime": "text/html",
                "format": "html",
                "caption": f"Borehole g-function #{run_no}",
                "kind": "report",
                "slot": f"borehole-gfunction-{run_no}",
            },
        )
        return "G-function plot opened in canvas."

    return plot_borehole_gfunction


def _borehole_temperatures_png(
    timestamps: list[str], T_in: list[float], T_out: list[float]
) -> str:
    """Render T_in / T_out vs time as a PNG and return it base64-encoded."""
    import base64
    import io
    from datetime import datetime

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    times = [datetime.fromisoformat(ts) for ts in timestamps]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, T_in, color="#d62728", linewidth=0.8, label="T_in")
    ax.plot(times, T_out, color="#1f77b4", linewidth=0.8, label="T_out")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Temperature [°C]")
    ax.set_title("Borehole fluid temperatures")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _borehole_gfunction_png(gfunc_time_s: list[float], gfunc_vals: list[float]) -> str:
    """Render g-function vs ln(t) as a PNG and return it base64-encoded."""
    import base64
    import io
    import math

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lnt = [math.log(t) for t in gfunc_time_s]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lnt, gfunc_vals, color="#2ca02c", linewidth=1.5)
    ax.set_xlabel("ln(t [s])")
    ax.set_ylabel("g")
    ax.set_title("G-function")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _make_view_borehole_temperatures_tool(session: Session):
    @tool
    async def view_borehole_temperatures() -> str | list[dict[str, Any]]:
        """View a T_in / T_out vs time plot from the most recent borehole simulation.

        Returns an image you can actually see, plus key statistics including
        monthly averages and freezing-risk hours. Call run_borehole_simulation first.
        Use plot_borehole_temperatures to show the plot to the user. 
        """
        from datetime import datetime

        result = _session_borehole_results.get(session.session_id)
        if result is None:
            return "No borehole simulation results available. Run run_borehole_simulation first."

        T_in: list[float] = result["T_in"]
        T_out: list[float] = result["T_out"]
        timestamps: list[str] = result["timestamps"]
        n = len(T_in)

        hours_below_0 = sum(1 for t in T_in if t < 0.0)
        hours_below_2 = sum(1 for t in T_in if t < 2.0)

        month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                       7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
        monthly_in: dict[tuple[int,int], list[float]] = {}
        monthly_out: dict[tuple[int,int], list[float]] = {}
        for ts_str, ti, to in zip(timestamps, T_in, T_out):
            try:
                dt = datetime.fromisoformat(ts_str)
                key = (dt.year, dt.month)
            except ValueError:
                continue
            monthly_in.setdefault(key, []).append(ti)
            monthly_out.setdefault(key, []).append(to)
        monthly_lines = [
            f"  {month_names[m]:3s} {y}: {sum(monthly_in[(y,m)])/len(monthly_in[(y,m)]):+.1f} / "
            f"{sum(monthly_out[(y,m)])/len(monthly_out[(y,m)]):+.1f} °C"
            for y, m in sorted(monthly_in)
        ]

        stats = (
            f"Run #{result['run_no']}, {n} hourly timesteps\n"
            f"T_in:  min {min(T_in):.2f} °C  max {max(T_in):.2f} °C  mean {sum(T_in)/n:.2f} °C\n"
            f"T_out: min {min(T_out):.2f} °C  max {max(T_out):.2f} °C  mean {sum(T_out)/n:.2f} °C\n"
            f"Hours T_in < 0 °C: {hours_below_0} ({100*hours_below_0/n:.1f}%)\n"
            f"Hours T_in < 2 °C: {hours_below_2} ({100*hours_below_2/n:.1f}%)\n"
            "Monthly mean T_in / T_out:\n" + "\n".join(monthly_lines)
        )

        try:
            b64 = _borehole_temperatures_png(timestamps, T_in, T_out)
        except Exception as exc:
            return f"Plot generation failed: {exc}\n\n{stats}"

        return [
            {"type": "text", "text": stats},
            {"type": "image", "mime_type": "image/png", "base64": b64},
        ]

    return view_borehole_temperatures


def _make_view_borehole_gfunction_tool(session: Session):
    @tool
    async def view_borehole_gfunction() -> str | list[dict[str, Any]]:
        """View a g-function vs ln(t) plot from the most recent borehole simulation.

        Returns an image you can actually see, plus key g values. Call
        run_borehole_simulation first. Use plot_borehole_gfunction to show the plot to the user. 
        """
        result = _session_borehole_results.get(session.session_id)
        if result is None:
            return "No borehole simulation results available. Run run_borehole_simulation first."

        import math

        gfunc_time_s: list[float] = result["gfunc_time"]
        gfunc_vals: list[float] = result["gfunc_vals"]
        lnt = [math.log(t) for t in gfunc_time_s]

        stats = "\n".join([
            f"Run #{result['run_no']}, {len(lnt)} data points",
            f"ln(t) range: {lnt[0]:.2f} to {lnt[-1]:.2f}",
            f"g range:     {min(gfunc_vals):.3f} to {max(gfunc_vals):.3f}",
        ])

        try:
            b64 = _borehole_gfunction_png(gfunc_time_s, gfunc_vals)
        except Exception as exc:
            return f"Plot generation failed: {exc}\n\n{stats}"

        return [
            {"type": "text", "text": stats},
            {"type": "image", "mime_type": "image/png", "base64": b64},
        ]

    return view_borehole_gfunction


def _fimbul_validation_png(
    T_in_C: list[float],
    T_out_C: list[float],
    durations_s: list[float],
    building_kWh: list[float],
    demand_kWh_windowed: list[float],
    COP: float,
    start_ts: str | None = None,
) -> str:
    """Two-panel PNG: (top) T_in/T_out vs time, (bottom) implied building heat vs demand."""
    import base64
    import io
    from datetime import datetime, timedelta

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    cum_s = [sum(durations_s[:i]) for i in range(len(durations_s))]
    if start_ts is not None:
        t0 = datetime.fromisoformat(start_ts)
        xs = [t0 + timedelta(seconds=s) for s in cum_s]
        use_dates = True
    else:
        xs = [s / 86400 for s in cum_s]
        use_dates = False

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1.plot(xs, T_in_C,  color="#d62728", linewidth=0.8, label="Tᵢₙ (prescribed)")
    ax1.plot(xs, T_out_C, color="#1f77b4", linewidth=0.8, label="Tₒᵤₜ (Fimbul)")
    ax1.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
    ax1.set_ylabel("Temperature [°C]")
    ax1.set_title("Fimbul validation: prescribed Tᵢₙ vs computed Tₒᵤₜ")
    ax1.legend()

    ax2.plot(xs, demand_kWh_windowed, color="#e67e22", linewidth=0.8, label="Heating needs")
    ax2.plot(xs, building_kWh,        color="#27ae60", linewidth=0.8,
             label="Fimbul implied building heating")
    ax2.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
    ax2.set_xlabel("Time" if use_dates else "Day of year")
    ax2.set_ylabel("Heating [kWh]")
    ax2.set_title("Fimbul implied building heat vs heating needs")
    ax2.legend()

    if use_dates:
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        fig.autofmt_xdate()

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _make_show_borehole_field_tool(session: Session):
    @tool
    async def show_borehole_field(  # noqa: PLR0913
        N_1: int = 1,
        N_2: int = 1,
        B: float = 5.0,
        H: float = 200.0,
        D: float = 0.5,
        r_b: float = 0.065,
        tilt: float = 0.0,
        orientation: float = 0.0,
        pattern: str = "rectangular",
        num_sectors: int | None = None,
        overrides: dict[str, dict[str, dict]] | None = None,
    ) -> str:
        """Visualize a borehole field on the geothermal map at the selected building's location.

        Shows each borehole as a 3D cylinder at its actual (x, y) position relative
        to the building, with depth H rendered as the cylinder height. This is the same
        visual style as clicking a well on the map. Only the field geometry matters here
        — thermal parameters (k_s, COP, etc.) are not used.

        The visualization replaces any previous field shown by this tool and disappears
        automatically when the user clicks another building or well on the map.

        Requires a building to be selected first (demand data must be available).

        Args:
            N_1: Number of boreholes in the x direction for pattern = "rectangular";
            else, total number of boreholes. Default 1.
            N_2: Number of boreholes in the y direction for pattern = "rectangular".
            If pattern = "polygonal", number of polygon sides. Default 1.
            B: Borehole centre-to-centre spacing [m]. Default 5.
            H: Active borehole depth [m]. Default 200.
            D: Buried depth to top of active section [m]. Default 0.5.
            r_b: Borehole radius [m]. Default 0.065.
            tilt: Borehole tilt from vertical [radians]. Default 0.
            orientation: Azimuthal direction of tilt [radians]. Default 0.
            pattern: Layout pattern — "rectangular", "sunflower", "circular", or
            "polygonal". Default "rectangular".
            num_sectors: Number of hydraulic sectors. Defaults to the total number
            of boreholes. Does not affect the visualization, but determines how
            overrides are indexed (sector then well within sector).
            overrides: Per-borehole parameter changes (same format as
            run_borehole_simulation). dx/dy shift a borehole relative to its
            pattern position; H overrides its individual display depth.
        """
        from backend_building_selection import get_session_demand_data
        from pygfunction_sim import (
            rectangle_field,
            sunflower_field,
            circular_field,
            polygonal_field,
        )

        demand = get_session_demand_data(session.session_id)
        if demand is None:
            return (
                "No building is selected. Click a building on the map and use "
                "'Generate energy demands' before calling this tool."
            )

        lat = demand.get("lat")
        lon = demand.get("lon")
        if lat is None or lon is None:
            return "Building location (lat/lon) is not available in the current demand data."

        if pattern == "rectangular":
            boreholes = rectangle_field(N_1, N_2, B, H, D, r_b, tilt, orientation, num_sectors)
        elif pattern == "sunflower":
            boreholes = sunflower_field(N_1, B, H, D, r_b, tilt, orientation, num_sectors)
        elif pattern == "circular":
            boreholes = circular_field(N_1, B, H, D, r_b, tilt, orientation, num_sectors)
        elif pattern == "polygonal":
            boreholes = polygonal_field(N_1, B, N_2, H, D, r_b, tilt, orientation, num_sectors)
        else:
            return (
                f"Unknown pattern {pattern!r}. "
                "Valid options: 'rectangular', 'sunflower', 'circular', 'polygonal'."
            )

        boreholes = _apply_overrides(boreholes, overrides)

        borehole_positions = [
            {
                "x": float(b.x),
                "y": float(b.y),
                "H": float(b.H),
                "D": float(b.D),
                "tilt": float(b.tilt),
                "orientation": float(b.orientation),
            }
            for sector in boreholes
            for b in sector
        ]
        n_total = len(borehole_positions)

        _ensure_map_pinned(session)
        session.trace.append(
            "ui",
            {
                "action": "show_borehole_field",
                "payload": {"lat": lat, "lon": lon, "boreholes": borehole_positions},
                "target": _MAP_TARGET,
            },
        )
        return f"Visualized {n_total} borehole(s) at the selected building ({lat:.5f}, {lon:.5f})."

    return show_borehole_field


def _make_run_fimbul_validation_tool(session: Session, simulation_jl_path: str):
    from jutul_agent.agent.tools import _capture_delta_writer

    @tool
    async def run_fimbul_validation(  # noqa: PLR0913
        N_1: int = 1,
        N_2: int = 1,
        B: float = 5.0,
        H: float = 200.0,
        D: float = 0.5,
        r_b: float = 0.065,
        tilt: float = 0.0,
        orientation: float = 0.0,
        k_s: float = 3.7,
        alpha: float = 1.59e-6,
        r_in: float = 0.015,
        r_out: float = 0.018,
        D_s: float = 0.030,
        k_p: float = 0.38,
        k_g: float = 2.3,
        epsilon: float = 1e-4,
        COP: float = 3.5,
        G: float = 0.03,
        pattern: str = "rectangular",
        num_sectors: int | None = None,
        window_hours: int = 24,
        overrides: dict[str, dict[str, dict]] | None = None,
    ) -> str:
        """Validate a pygfunction borehole geometry with a Fimbul PDE simulation.

        Runs pygfunction with the given parameters (identical to run_borehole_simulation)
        to obtain T_in, aggregates it into windows of `window_hours` hours (default:
        daily), then prescribes each window's mean T_in as the Fimbul injection
        temperature. Returns T_out and net extracted heat energy per window.

        All parameters must match the run_borehole_simulation call being validated
        to ensure a faithful comparison. The same overrides are applied to both the
        pygfunction run and the Fimbul field, keeping both sides in sync.
        Use view_fimbul_validation to plot.

        Args:
            N_1: Number of boreholes in x (rectangular) or total boreholes. Default 1.
            N_2: Number of boreholes in y (rectangular) or polygon sides. Default 1.
            B: Borehole spacing [m] or sunflower radius [m]. Default 5.
            H: Active borehole depth [m]. Default 200.
            D: Buried depth [m]. Default 0.5 (matches Fimbul).
            r_b: Borehole radius [m]. Default 0.065 (matches Fimbul).
            tilt: Tilt angle from vertical [radians]. Default 0.
            orientation: Azimuthal direction of tilt [radians]. Default 0.
            k_s: Ground thermal conductivity [W/(m·K)]. Default 3.7 (matches Fimbul).
            alpha: Ground thermal diffusivity [m²/s]. Default 1.59e-6 (matches Fimbul: 3.7/(2580×900)).
            r_in: Pipe inner radius [m]. Default 0.015.
            r_out: Pipe outer radius [m]. Default 0.018 (matches Fimbul).
            D_s: Shank spacing [m]. Default 0.030 (matches Fimbul).
            k_p: Pipe wall thermal conductivity [W/(m·K)]. Default 0.38 (matches Fimbul).
            k_g: Grout thermal conductivity [W/(m·K)]. Default 2.3 (matches Fimbul).
            epsilon: Pipe roughness [m]. Default 1e-4 (matches Fimbul/JutulDarcy).
            COP: Heat-pump COP. Default 3.5.
            G: Geothermal gradient [K/m]. Default 0.03 (matches Fimbul).
            pattern: Field layout — "rectangular", "sunflower", "circular",
                or "polygonal". Default "rectangular".
            num_sectors: Number of hydraulic sectors. Boreholes within a sector are
                connected in series; sectors are connected in parallel. Defaults to the
                total number of boreholes (one borehole per sector = fully parallel field).
            window_hours: Averaging window width in hours. Default 24 (daily).
            overrides: Optional per-borehole parameter changes applied after the
            field is generated from the pattern. Structure:
            {"sector_index": {"well_index": {param: value, ...}, ...}, ...}
            where sector_index and well_index are 0-based string integers. Within
            each borehole dict the following keys are recognised:
              H (float): active borehole depth [m].
              D (float): buried depth to top of active section [m].
              r_b (float): borehole radius [m].
              tilt (float): tilt from vertical [radians].
              orientation (float): azimuthal direction of tilt [radians].
              dx (float): displacement added to the pattern x-coordinate [m].
              dy (float): displacement added to the pattern y-coordinate [m].
            dx/dy shift a borehole relative to its pattern-generated position;
            all other keys replace the uniform value directly.
            Example — deepen well 1 in sector 0 and shift well 2 by (2, 3) m:
            {"0": {"1": {"H": 250.0}, "2": {"dx": 2.0, "dy": 3.0}}}
        """
        import asyncio

        import pandas as pd
        import pygfunction as gt

        from backend_building_selection import get_session_demand_data
        from pygfunction_sim import (
            FluidParams,
            GroundParams,
            PipeParams,
            boreholes_to_fimbul_field,
            circular_field,
            fetch_mean_surface_temperature,
            polygonal_field,
            rectangle_field,
            simulate_borehole_temperatures,
            sunflower_field,
        )

        demand = get_session_demand_data(session.session_id)
        if demand is None:
            return "No demand data found. Click 'Generate energy demands' first."

        df = pd.DataFrame({"time": demand["timestamps"], "kW": demand["demand_kw"]})
        if pattern == "rectangular":
            boreholes = rectangle_field(N_1, N_2, B, H, D, r_b, tilt, orientation, num_sectors)
        elif pattern == "sunflower":
            boreholes = sunflower_field(N_1, B, H, D, r_b, tilt, orientation, num_sectors)
        elif pattern == "circular":
            boreholes = circular_field(N_1, B, H, D, r_b, tilt, orientation, num_sectors)
        elif pattern == "polygonal":
            boreholes = polygonal_field(N_1, B, N_2, H, D, r_b, tilt, orientation, num_sectors)
        else:
            return f"Unknown pattern {pattern!r}."
        boreholes = _apply_overrides(boreholes, overrides)

        try:
            df_result, _, _ = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: simulate_borehole_temperatures(
                    df,
                    lat=demand.get("lat"),
                    lon=demand.get("lon"),
                    boreholes=boreholes,
                    ground=GroundParams(k_s=k_s, alpha=alpha),
                    pipe=PipeParams(r_in=r_in, r_out=r_out, D_s=D_s, k_p=k_p,
                                    k_g=k_g, epsilon=epsilon),
                    COP=COP,
                    G=G,
                ),
            )
        except Exception as exc:
            return f"pygfunction failed: {exc}"

        T_in_avg, durations_s = _aggregate_tin_profile(
            df_result["T_in"].tolist(), window_hours=window_hours
        )

        T_surface = fetch_mean_surface_temperature(demand["lat"], demand["lon"])
        fluid = FluidParams()
        fl = gt.media.Fluid(fluid.fluid_name, fluid.concentration)
        freeze_point_C = fl.fluid.t_min

        below_freeze_days = [i + 1 for i, t in enumerate(T_in_avg) if t < freeze_point_C]
        if below_freeze_days:
            if len(below_freeze_days) <= 30:
                days_str = ", ".join(map(str, below_freeze_days))
            else:
                days_str = (
                    f"{below_freeze_days[0]}–{below_freeze_days[-1]} "
                    f"({len(below_freeze_days)} days total)"
                )
            return (
                f"Cannot run Fimbul validation: the prescribed inlet temperature "
                f"drops below the freezing point of the working fluid "
                f"({freeze_point_C:.1f} °C for {fluid.fluid_name}) "
                f"on {len(below_freeze_days)} day(s): {days_str}. "
                f"This is a physically unrealisable configuration — the fluid would "
                f"freeze in the boreholes. Consider increasing the number of boreholes, "
                f"reducing the depth, or lowering the heating demand."
            )

        setup = {
            "field": boreholes_to_fimbul_field(boreholes),
            "k_s":                 k_s,
            "alpha":               alpha,
            "G":                   G,
            "T_surface_C":         T_surface,
            "m_flow_per_borehole": fluid.m_flow_per_borehole,
            "rho_fluid":           fl.rho,
            "cp_fluid":            fl.cp,
            "T_in_C":              T_in_avg,
            "durations_s":         durations_s,
        }

        result_path = (
            session.output_dir / "artifacts" /
            f"val-result-{uuid.uuid4().hex[:8]}.json"
        )
        result_path.parent.mkdir(parents=True, exist_ok=True)
        code = _render_template(
            _BTES_VALIDATION_TEMPLATE,
            JL_PATH=Path(simulation_jl_path).resolve().as_posix(),
            SETUP_JSON=json.dumps(setup),
            RESULT_PATH=result_path.as_posix(),
        )

        jl_result = await session.julia.eval(code, on_chunk=_capture_delta_writer())
        if jl_result.error:
            return f"Julia error: {jl_result.error}"

        val = json.loads(result_path.read_text(encoding="utf-8"))
        if val.get("status") != "completed":
            return f"Fimbul error: {val.get('message', 'unknown')}"

        ground_kWh = val["energy_kWh"]
        building_kWh = [e * COP / (COP - 1.0) for e in ground_kWh]

        _session_fimbul_validation_results[session.session_id] = {
            "window_hours":   window_hours,
            "T_in_C":         val["T_in_C"],
            "T_out_C":        val["T_out_C"],
            "energy_kWh":     ground_kWh,
            "building_kWh":   building_kWh,
            "durations_s":    val["durations_s"],
            "n_periods":      val["n_periods"],
            "COP":            COP,
        }

        T_out = val["T_out_C"]
        n = val["n_periods"]
        total_building = sum(building_kWh)
        return (
            f"Fimbul validation: {n} windows of {window_hours} h.\n"
            f"T_out: min {min(T_out):.1f} °C  max {max(T_out):.1f} °C  "
            f"mean {sum(T_out)/n:.1f} °C\n"
            f"Total implied building heat: {total_building:.0f} kWh "
            f"(ground extraction × COP/(COP−1), COP={COP})\n"
            "Use view_fimbul_validation to plot."
        )

    return run_fimbul_validation


def _make_view_fimbul_validation_tool(session: Session):
    @tool
    async def view_fimbul_validation() -> str | list[dict[str, Any]]:
        """Open a canvas plot and show the image and statistics from the most recent Fimbul validation.

        Opens a canvas tab with (1) prescribed T_in vs Fimbul-computed T_out and
        (2) Fimbul heat extraction vs windowed heating demand. Also returns the
        image and key statistics (total extraction, total demand, temperature
        ranges) directly so they are visible here in the chat.
        Call run_fimbul_validation first.
        """
        from backend_building_selection import get_session_demand_data

        result = _session_fimbul_validation_results.get(session.session_id)
        if result is None:
            return "No Fimbul validation result found. Run run_fimbul_validation first."

        T_in_C       = result["T_in_C"]
        T_out_C      = result["T_out_C"]
        energy_kWh   = result["energy_kWh"]
        building_kWh = result["building_kWh"]
        durations_s  = result["durations_s"]
        n            = result["n_periods"]
        wh           = result["window_hours"]
        COP          = result["COP"]

        # Aggregate raw demand (kW, hourly) over the same windows used by Fimbul.
        demand = get_session_demand_data(session.session_id)
        if demand is not None:
            raw_kw: list[float] = demand["demand_kw"]
            demand_kWh_windowed: list[float] = []
            i = 0
            for dur in durations_s:
                w = round(dur / 3600)
                chunk = raw_kw[i : i + w]
                demand_kWh_windowed.append(float(sum(chunk)))
                i += len(chunk)
        else:
            demand_kWh_windowed = [0.0] * n

        start_ts = demand["timestamps"][0] if demand is not None and demand.get("timestamps") else None
        try:
            b64 = _fimbul_validation_png(
                T_in_C, T_out_C, durations_s, building_kWh, demand_kWh_windowed, COP, start_ts
            )
        except Exception as exc:
            return f"Plot generation failed: {exc}"

        html = (
            "<!doctype html><html><head><title>Fimbul validation</title>"
            "<style>body{margin:0;background:#fff}img{max-width:100%;display:block}</style>"
            "</head><body>"
            f'<img src="data:image/png;base64,{b64}">'
            "</body></html>"
        )
        run_no = _fimbul_validation_run_count.get(session.session_id, 0) + 1
        _fimbul_validation_run_count[session.session_id] = run_no
        rel = f"artifacts/fimbul-validation-{run_no}.html"
        out = session.output_dir / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        session.trace.append(
            "artifact",
            {
                "path": rel,
                "mime": "text/html",
                "format": "html",
                "caption": f"Fimbul validation #{run_no}",
                "kind": "report",
                "slot": f"fimbul-validation-{run_no}",
            },
        )

        import math
        diffs = [b - d for b, d in zip(building_kWh, demand_kWh_windowed)]
        mean_diff = sum(diffs) / n
        rmse = math.sqrt(sum(d ** 2 for d in diffs) / n)

        stats = (
            f"{n} time windows of {wh} h  (COP={COP})\n"
            f"T_in:  min {min(T_in_C):.2f} °C  max {max(T_in_C):.2f} °C  "
            f"mean {sum(T_in_C)/n:.2f} °C\n"
            f"T_out: min {min(T_out_C):.2f} °C  max {max(T_out_C):.2f} °C  "
            f"mean {sum(T_out_C)/n:.2f} °C\n"
            f"Total implied building heat: {sum(building_kWh):.0f} kWh\n"
            f"Total heating demand:        {sum(demand_kWh_windowed):.0f} kWh\n"
            f"Mean difference (Fimbul − needs): {mean_diff:+.2f} kWh/window\n"
            f"RMSE (Fimbul vs needs):           {rmse:.2f} kWh/window\n"
            f"Time windows with negative extraction: {sum(1 for e in energy_kWh if e < 0)}"
        )
        return [
            {"type": "text", "text": stats},
            {"type": "image", "mime_type": "image/png", "base64": b64},
        ]

    return view_fimbul_validation


def _make_go_to_building_tool(session: Session):
    @tool
    async def go_to_building(bygningsnummer: str) -> str:
        """Fly the geothermal map to a specific building and select it.

        Looks up the building by its Norwegian building number (bygningsnummer)
        in the Matrikkelen WFS and moves the map to its location, exactly as if
        the user had clicked it on the map.

        Args:
            bygningsnummer: The Norwegian building number to look up (e.g. "12345678").
        """
        import asyncio as _asyncio

        from backend_building_selection import fetch_building_by_number

        try:
            building = await _asyncio.get_event_loop().run_in_executor(
                None, fetch_building_by_number, bygningsnummer
            )
        except Exception as exc:
            return f"Could not look up building {bygningsnummer}: {exc}"

        if building is None:
            return f"No building with number '{bygningsnummer}' found in Matrikkelen."

        _ensure_map_pinned(session)
        session.trace.append(
            "ui",
            {
                "action": "select_building",
                "payload": building.model_dump(),
                "target": _MAP_TARGET,
            },
        )
        parts = [f"Building {building.bygningsnummer}"]
        if building.bygningstype:
            parts.append(building.bygningstype)
        if building.kommunenavn:
            parts.append(building.kommunenavn)
        return f"Moved the map to {', '.join(parts)} ({building.lat:.5f}, {building.lon:.5f})."

    return go_to_building


_PROMPT_FRAGMENT = (
    "This app shows a map of Norwegian borehole data next to the chat (it "
    "appears the first time you use one of the tools below). Call "
    "`set_map_view` to fly it to a raw location, `go_to_address` to fly to a "
    "Norwegian street address (it resolves the address via Kartverket and moves "
    "the map automatically — use this whenever the user gives an address rather "
    "than a well number or coordinates), `go_to_building` to fly to and select "
    "a specific building by its Norwegian building number (bygningsnummer) — it "
    "queries Matrikkelen directly and selects the building exactly as if the user "
    "had clicked it, `go_to_well` to fly to and "
    "select a specific well by its number or other identifying text, or "
    "`go_to_well_park` to do the same for a well park itself (e.g. when asked "
    "about a BTES site as a whole, not one of its individual wells) — both "
    "tell you directly if no such well/park exists, so trust their return "
    "value rather than assuming success. The user can also click things on "
    "the map themselves (e.g. selecting a well or building); when they do, a note "
    "describing it is prepended to their next message as '[UI events since "
    "your last message]', so you'll see it as part of what they sent.\n\n"
    "Call `run_simulation` to run a Fimbul geothermal simulation (an AGS or "
    "BTES case) with given parameters, and `view_simulation_result` to look at "
    "a reservoir-state image from the most recent run. The user's own map "
    "sidebar also has a 'Setup Simulation' button that resolves and runs these "
    "directly (bypassing you) for a selected well — if they used it, you'll "
    "learn about the completed run the same way as any other UI event.\n\n"
    "Call `show_borehole_field` to visualize a borehole field on the map at the "
    "selected building's location, showing each borehole as a 3D cylinder at its "
    "actual position. Use it whenever the user wants to see a field layout — "
    "before or after running a simulation, or to compare different configurations. "
    "It takes the same geometry parameters as `run_borehole_simulation` and `run_fimbul_validation`. "
    "The visualization disappears when the user clicks another building or well.\n\n"
    "Call `run_borehole_simulation` to run a single pygfunction borehole heat "
    "exchanger simulation on the heating demands most recently generated via the "
    "'Generate energy demands' button. Use this for a single configuration. "
    "After a successful simulation, use `plot_borehole_temperatures` or "
    "`plot_borehole_gfunction` to show/visualize a T_in/T_out vs time plot or a g-function vs time plot, " \
    "respectively. The charts are displayed in the canvas tab. Use `view_borehole_temperatures` to view the "
    "T_in/T_out vs time plot yourself (returns an image you can read and describe, "
    "plus monthly averages and freezing-risk statistics). Simlarly, `view_borehole_gfunction` "
    "allows you to view and read the g-function vs ln(t) plot. \n\n"
    "Call `sweep_borehole_parameters` only when the user explicitly asks to "
    "explore or optimise over multiple borehole configurations at once (e.g. "
    "varying depth, number of boreholes, or ground properties). Each parameter "
    "accepts a list of values; the tool runs one simulation per element of the "
    "Cartesian product and returns min/mean/max T_in and T_out for every "
    "combination. Keep individual lists short — the total number of simulations "
    "is the product of all list lengths. After the sweep, call "
    "`run_borehole_simulation` with the best combination if the user wants to "
    "inspect or plot that configuration.\n\n"
    "Call `run_fimbul_validation` to validate a borehole geometry with a full "
    "Fimbul PDE simulation. It runs pygfunction internally with the given "
    "parameters to obtain T_in, aggregates it into daily windows (configurable "
    "via `window_hours`), prescribes each window's mean T_in as the Fimbul "
    "injection temperature, and returns T_out and net extracted energy per "
    "window. Pass exactly the same parameters as the `run_borehole_simulation` "
    "call being validated (N_1, N_2, B, H, D, r_b, k_s, G, pattern, etc.) so "
    "the comparison is faithful. Call `view_fimbul_validation` afterwards — it "
    "opens a two-panel canvas plot (T_in/T_out temperatures and Fimbul "
    "extraction vs heating demand) and returns both the plot image and summary "
    "statistics as text, so you can see and reason about the results directly."

)


def geothermal_map_capability(
    data_path: str,
    simulation_jl_path: str,
    pygsim_jl_path: str | None = None,
) -> Capability:
    """The web capability for the geothermal map: well-lookup and simulation
    tools that drive the native ``map`` canvas panel (see
    ``canvas/MapPanel.tsx``).

    ``data_path`` is the absolute path to the borehole GeoJSON file;
    ``simulation_jl_path`` is the absolute path to ``julia/simulation.jl``.
    ``pygsim_jl_path`` is accepted for backwards compatibility but no longer used.
    """
    return Capability(
        name="geothermal-map",
        tools=(
            _make_set_map_view_tool,
            _make_go_to_address_tool,
            _make_go_to_building_tool,
            lambda session: _make_go_to_well_tool(session, data_path),
            lambda session: _make_go_to_well_park_tool(session, data_path),
            lambda session: _make_run_simulation_tool(session, simulation_jl_path),
            lambda session: _make_view_simulation_result_tool(session, simulation_jl_path),
            _make_run_borehole_simulation_tool,
            _make_show_borehole_field_tool,
            _make_sweep_borehole_parameters_tool,
            _make_plot_borehole_temperatures_tool,
            _make_plot_borehole_gfunction_tool,
            _make_view_borehole_temperatures_tool,
            _make_view_borehole_gfunction_tool,
            lambda session: _make_run_fimbul_validation_tool(session, simulation_jl_path),
            _make_view_fimbul_validation_tool,
        ),
        prompt_fragment=_PROMPT_FRAGMENT,
        surfaces=("web",),
        on_connect=(_ensure_map_pinned,),
    )
