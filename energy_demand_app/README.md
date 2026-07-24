# jutul-agent geothermal map

A geothermal map of Norwegian borehole data, shown as a native panel in
jutul-agent's own canvas (`canvas/MapPanel.tsx`) — not a separate page in an
iframe, and not backed by a second, dedicated Julia kernel. The map runs in
the same process, and Fimbul simulations (once wired up — see below) run on
the same Julia kernel as the rest of the chat session.

This is the successor to `examples/geothermal-viz-app`, ported on top of the
canvas extension seam described in
[docs/web-ui.md](../../docs/web-ui.md#extending-the-canvas). `geothermal-viz`
(the sibling repo this was ported from) remains the reference for the map's
rendering and Fimbul logic — nothing here changes it.

What it demonstrates:

- **A native canvas panel.** `MapPanel.tsx`, registered as the `"map"` kind in
  jutul-agent's own webapp, renders the borehole layers, popups, and well info
  directly — no iframe, no `postMessage` bridge.
- **Well-lookup tools, resolved server-side.** `go_to_well`/`go_to_well_park`
  look the well up against the same GeoJSON file the map renders from (read
  straight off disk, in-process), so the agent gets an honest found/not-found
  answer in the same turn, rather than waiting on the browser to report back.
- **Targeted `ui` actions.** Each tool emits a `ui` trace event aimed at the
  map's own canvas view (`target="slot:geothermal-map"`) instead of a global
  one — see `capability.py`'s `_MAP_TARGET` and `MapPanel.tsx`'s
  `useUiActions` hook.

Not yet wired up: running a Fimbul simulation from the map (the old app's
"Setup Simulation" sidebar). That's a later phase — see `julia/simulation.jl`,
copied over from `geothermal-viz` but not yet `include()`d by anything here.

## Run it

This runs on jutul-agent's built-in `fimbul` simulator, so no extra Julia
environment is needed beyond what that simulator already declares:

```bash
python examples/geothermal-map/serve.py
```

Open <http://127.0.0.1:8740>. Try: *"go to well 12345"*, *"fly to well park ...
"*, or *"move the map to longitude 10.7, latitude 59.9, zoom 12"*. The map
panel appears in the canvas the first time you ask for one of these.

## Layout

- `data/all_boreholes.geojson` — the borehole dataset, copied from
  `geothermal-viz/processed_data/`. `MapPanel.tsx` fetches it directly from
  `/geothermal-data/all_boreholes.geojson` (see `serve.py`'s `extra_mounts`);
  `capability.py` reads the same file straight off disk for well lookups.
- `julia/simulation.jl` — copied verbatim from `geothermal-viz/src/`, for a
  later phase to wire a `run_simulation` tool through.
- `scripts/process_data.jl` — copied verbatim; an offline, one-time step (not
  part of this app's runtime) that regenerates `data/all_boreholes.geojson`
  from the source geodatabase. See `geothermal-viz/data/README.md`.
- `capability.py` — the well-lookup tools (`set_map_view`, `go_to_well`,
  `go_to_well_park`) and `geothermal_map_capability()`.
- `serve.py` — the server wiring: a `fimbul` session with the capability added,
  plus the static mount for the borehole data.

## Extending it

- Wire up `run_simulation`/`view_simulation_result` against `julia/simulation.jl`
  the same way `examples/geothermal-viz-app/capability.py` did, but against the
  session's own kernel instead of a dedicated one.
- Add tools, skills, or subagents by extending the `Capability` in
  `capability.py`, or publish your own under the `jutul_agent.extensions` entry
  point.
