# Geoagent

[![CI](https://github.com/fredeco-gh/jutul-agent-geothermal/actions/workflows/ci.yml/badge.svg)](https://github.com/fredeco-gh/jutul-agent-geothermal/actions/workflows/ci.yml)
[![Simulators](https://github.com/fredeco-gh/jutul-agent-geothermal/actions/workflows/simulators.yml/badge.svg)](https://github.com/fredeco-gh/jutul-agent-geothermal/actions/workflows/simulators.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An AI-powered geothermal planning tool for Norwegian borehole fields and building energy systems. Geoagent combines an interactive map of Norwegian borehole data, building heating-demand estimation, physics-based borehole simulation, and full Fimbul PDE reservoir simulation — all driven by a conversational AI agent or by direct UI controls.

Geoagent is developed on top of [jutul-agent](https://github.com/SINTEF-agentlab/jutul-agent), the open-source scientific AI agent framework by [SINTEF](https://www.sintef.no/en/). The interactive map is based on GeothermalViz, also developed by SINTEF.

<p align="center">
  <img src="docs/images/web-fimbul.png" alt="Geoagent: Fimbul well simulation with 3D temperature field" width="47%">
  &nbsp;&nbsp;&nbsp;
  <img src="docs/images/web-btes.png" alt="Geoagent: BTES well park simulation results" width="47%">
</p>
<p align="center"><sub>Left: an AGS doublet simulation with a 3D temperature field rendered in the canvas. Right: a BTES well park with well output time series.</sub></p>

---

## Contents

- [Accessing the app](#accessing-the-app)
- [Features](#features)
- [Agent tools](#agent-tools)
- [Architecture](#architecture)
- [Development](#development)

---

## Accessing the app

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Julia
(via [juliaup](https://github.com/JuliaLang/juliaup)). Then:

```sh
uv tool install "git+https://github.com/fredeco-gh/jutul-agent-geothermal"
geoagent key openai      # save a provider key (Anthropic and Google work too)

geoagent init            # precompiles Julia/Fimbul once (~10 min)
geoagent                 # open http://127.0.0.1:8740
```

Run from any folder — no clone needed. Local models through [Ollama](https://ollama.com)
need no key. `init` only needs to run once; after that `geoagent` starts in seconds.
`jutul-agent doctor` checks the setup if anything looks wrong.
Update with `uv tool install --reinstall "git+https://github.com/fredeco-gh/jutul-agent-geothermal"`.

The app opens at <http://127.0.0.1:8740> and runs locally for a single trusted user.

**API keys.** Place a `.env` file anywhere above your working directory with the key for your chosen LLM provider:

```
OPENAI_API_KEY=sk-...
# or
ANTHROPIC_API_KEY=sk-ant-...
```

The server walks up the directory tree from its install location to find the nearest `.env`.

---

## Features

### Interactive borehole map

The app opens with an interactive map of Norwegian borehole data sourced from the national registry. Click any marker to select a well or well park; the agent sees the selection immediately and can look up parameters, navigate to the site, or kick off a simulation. You can also do a manual simulation. Click "Setup simulation" to open a sidepanel to type in simulation parameters. Note that it may take a few minutes to load the setup the first time each session. 

### Building energy needs

Click a building on the map to select it, then click **Analyze heating needs** in the sidebar to open the heating panel. There you can configure the simulation year, building type, usable floor area, energy efficiency class, and demand type (space heating, domestic hot water, or combined). Click **Generate heating needs** to run the computation.

The app is designed to fetch real hourly demand profiles from the [PROFet API](https://www.sintef.no/en/projects/2019/profet/) — a statistical model of Norwegian building energy use. Ground temperature is derived from ERA5-Land climate normals (1991–2020) at the building's coordinates. The generated profile is then available to the agent's borehole simulation tools.

> **Note:** PROFet requires an API key that is not included in this repository. Without it, the app falls back to a synthetic heating demand timeseries so the rest of the workflow (borehole simulation, Fimbul validation) can still be exercised. To enable real building energy profiles, add a PROFet API key to your `.env` file once access has been arranged.

### Borehole field simulation (pygfunction)

Given a building's demand profile, the agent can design a borehole heat exchanger field — specifying the number of boreholes, depth, spacing, pattern, and thermal properties — and run a [pygfunction](https://github.com/MassimoCimmino/pygfunction) g-function simulation to predict hourly carrier-fluid temperatures (T_in and T_out) over the full demand period. The agent warns when T_in drops below 0 °C (freeze risk). Pattern options are rectangular, sunflower, circular, and polygonal, with optional per-borehole overrides for geometry.

### Fimbul reservoir simulation

Wells and well parks on the map can be simulated directly with [Fimbul.jl](https://github.com/fredeco-gh/Fimbul), a physics-based PDE simulator for geothermal systems built on [JutulDarcy.jl](https://github.com/sintefmath/JutulDarcy.jl). Click a well to auto-populate simulation parameters from its metadata, then click **Run simulation** in the sidebar or ask the agent to run it. Supported configurations:

| Case type | Description |
|---|---|
| AGS | Closed-loop single-well Advanced Geothermal System (coaxial borehole) |
| BTES | Open-loop Borehole Thermal Energy Storage well park (multiple wells, seasonal charge/discharge) |

Results are shown in an interactive report tab: well output time series (temperature, flow rate) and scrubbable 3D reservoir-state images (temperature field, pressure).

### Fimbul validation

The agent can cross-validate a pygfunction design against a full Fimbul PDE simulation. It runs both models on the same geometry and demand profile, then compares the implied T_out and heat extraction side by side. This gives a physics-grounded check on the g-function approximation before committing to a field design.

### Recommended workflow for building sites

The intended end-to-end workflow for finding and validating an optimal borehole field for a building is:

1. **Select a building** on the map and generate its heating demand profile (**Analyze heating needs** → **Generate heating needs**).
2. **Explore geometries with pygfunction.** Ask the agent to sweep across candidate field layouts — varying the number of boreholes, depth, spacing, and pattern — using `sweep_borehole_parameters`. pygfunction evaluates each combination quickly via g-functions, making it practical to sample a large design space in a single step.
3. **Identify promising candidates.** The agent reviews the sweep results (minimum T_in, mean T_out, freeze risk) and narrows the field down to a shortlist of geometries that meet thermal and practical constraints.
4. **Validate with Fimbul.** For each candidate, the agent runs a Fimbul validation to check the simplified g-function result against a full PDE reservoir simulation. This confirms whether the design holds up under real subsurface physics before any commitment is made.

The split between pygfunction (fast, approximate) and Fimbul (slow, physics-accurate) is deliberate: pygfunction makes broad exploration feasible, while Fimbul provides the rigour needed for a final design decision.

---

## Agent tools

The agent has the following extra tools at its disposal in addition to the same capabilities as jutul-agent. They are wired in through `energy_demand_app/capability.py` and exposed automatically at session start. This list may be useful to get a sense for what the agent can do. 

### Map navigation

| Tool | Description |
|---|---|
| `set_map_view(lon, lat, zoom)` | Pan and zoom the map to any coordinates. |
| `go_to_address(address, zoom)` | Fly to a Norwegian street address. Resolves the address via Kartverket's Matrikkelen API and moves the map automatically. |
| `go_to_well(identifier)` | Fly to and select a borehole by well number, area name, contractor, or other identifying text. Searches the local borehole dataset. Reports honestly if no match is found. |
| `go_to_well_park(identifier)` | Fly to and select a well park (e.g. a BTES site) by park number or name. Searches the local borehole dataset; restricted to park-level features so a park search can't accidentally land on an individual well. |
| `go_to_building(bygningsnummer)` | Fly to and select a building by its Norwegian building number. Queries Matrikkelen directly, identical to a map click. |

### Well simulation (Fimbul — sidebar integration)

These tools work alongside the sidebar's **Setup Simulation** and **Run simulation** buttons. The buttons bypass the model for pure data lookups and run the simulation directly with the form's parameters; the tools let the agent do the same conversationally.

| Tool | Description |
|---|---|
| `get_selected_well_params()` | Return the Fimbul simulation parameters resolved from the most recently clicked well or well park. Includes `case_type` (AGS or BTES) and all physical parameters derived from the well's registry metadata. Use these as a starting point when the user asks to simulate a specific well. |
| `view_simulation_result(var, step, delta)` | Inspect a variable from the most recent Fimbul simulation run via the sidebar. For reservoir state variables (e.g. `Temperature`, `Pressure`) renders a 3D image of the spatial field. For well output variables (e.g. `temperature`, `AqueousMassRate`) returns a time series. Pass `delta=True` to show change from the initial state rather than the absolute value. |

### Borehole design (pygfunction)

These tools require energy demand data to be available (generated via the sidebar's **Generate energy demands** button for a selected building).

| Tool | Description |
|---|---|
| `run_borehole_simulation(N_1, N_2, B, H, D, r_b, k_s, k_g, COP, pattern, …, overrides)` | Run a pygfunction U-tube borehole heat exchanger simulation. Computes hourly carrier-fluid temperatures T_in and T_out for a borehole field with the given geometry and thermal parameters. Pattern options: `rectangular`, `sunflower`, `circular`, `polygonal`. Warns when daily-average T_in falls below the freezing point of the fluid. It also includes a parameter 'overrides', which allows the agent to override the geometric properties of individual boreholes - such as if the user wants to shift one or more of the boreholes relative to the others, or change its length, tilt, orientation etc. |
| `sweep_borehole_parameters(N_1, N_2, B, H, …, overrides)` | Run pygfunction for every combination of the provided parameter lists (Cartesian product). Returns summary statistics (min/mean/max T_in and T_out) for each combination. Use this when the user asks to compare multiple borehole layouts or find an optimal configuration. This tool also includes the overrides parameter. |
| `show_borehole_field(N_1, N_2, B, H, pattern, …)` | Visualise a proposed borehole field as 3D cylinders on the map at the selected building's location. Uses the field geometry only — thermal parameters have no effect on the visualisation. |

### Borehole results

| Tool | Description |
|---|---|
| `plot_borehole_temperatures()` | Open a canvas tab with a T_in / T_out vs time chart from the most recent borehole simulation. |
| `plot_borehole_gfunction()` | Open a canvas tab with the g-function vs ln(t) curve from the most recent borehole simulation. |
| `view_borehole_temperatures()` | Render the T_in / T_out chart and return it as an image directly in the chat. Use `plot_borehole_temperatures` if a persistent canvas tab is preferred. |
| `view_borehole_gfunction()` | Render the g-function chart and return it as an image in the chat. |

### Fimbul validation

| Tool | Description |
|---|---|
| `run_fimbul_validation(N_1, N_2, B, H, …, window_hours)` | Cross-validate a pygfunction borehole geometry with a Fimbul PDE simulation. Runs pygfunction to obtain T_in, aggregates it into windows of `window_hours` hours (default: 24 = daily), then runs Fimbul with each window's mean T_in prescribed as the injection temperature. Returns T_out and net heat extraction per window alongside the building's demand, making it easy to check how well the simplified g-function model matches the full physics. Parameters must match a prior `run_borehole_simulation` call to ensure a fair comparison. |
| `view_fimbul_validation()` | Open a canvas tab and return an image showing (1) prescribed T_in vs Fimbul-computed T_out and (2) Fimbul heat extraction vs windowed building demand. Also reports total extraction, total demand, RMSE, and any windows with negative extraction. |

---

## Architecture

Geoagent is built as an extension of [jutul-agent](https://sintef-agentlab.github.io/jutul-agent/), the open-source scientific AI agent framework developed at SINTEF. Key components:

```
energy_demand_app/
├── serve.py               # Server entry point; wires the capability into jutul-agent
├── capability.py          # All agent tools and sidebar action handlers
├── backend_building_selection.py  # Building lookup, PROFet demand generation
├── pygfunction_sim.py     # Borehole field geometry and g-function simulation helpers
├── energy_demand.py       # Demand aggregation utilities
├── era5_land_timeseries.py # ERA5-Land climate normal lookup
├── julia/
│   └── simulation.jl      # Fimbul simulation runner (AGS, BTES) — executed in the
│                          #   session's own persistent Julia kernel via PythonCall
└── data/
    └── all_boreholes.geojson  # Norwegian borehole dataset (regenerated offline by
                               #   scripts/process_data.jl from the source geodatabase)
```

The map panel (`canvas/MapPanel.tsx`) is registered as a native canvas panel in jutul-agent's web UI — not an iframe — so the map and the chat share a single process, a single Julia kernel, and a single session. There is no dedicated Julia server: borehole data is read from disk in-process, and Fimbul simulations run on the same kernel as the rest of the chat.

**Data sources**

| Source | Used for |
|---|---|
| [Matrikkelen / Kartverket](https://www.kartverket.no/) | Building lookups; address geocoding |
| [PROFet API](https://www.sintef.no/en/projects/2019/profet/) | Hourly building heating demand profiles |
| [ERA5-Land](https://cds.climate.copernicus.eu/) | Climate normals (1991–2020) for undisturbed ground temperature |
| [Fimbul.jl](https://github.com/sintefmath/Fimbul.jl) | PDE reservoir simulation (AGS, BTES) |
| [pygfunction](https://github.com/MassimoCimmino/pygfunction) | Borehole g-function and U-tube heat exchanger simulation |

---

## Development

Work from a clone and run through `uv`:

```sh
git clone <this-repo>
cd jutul-agent-geothermal
uv sync --extra eval
uv run pre-commit install

uv run ruff check .          # lint
uv run pytest                # unit tests
uv run pytest -m integration # adds Julia-requiring tests
```

To run the app from the clone (without installing):

```sh
uv tool install .            # installs geoagent and jutul-agent CLIs from the local clone
geoagent init                # precompiles Julia/Fimbul into ~/.geoagent/workspace/
geoagent                     # open http://127.0.0.1:8740
```

To regenerate the borehole dataset from the source geodatabase:

```sh
julia energy_demand_app/scripts/process_data.jl
```

This is an offline, one-time step that writes `energy_demand_app/data/all_boreholes.geojson`. The script is not part of the app's runtime.

Developed at [SINTEF](https://www.sintef.no/en/).