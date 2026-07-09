"""
    Simulation parameter mapping layer

Maps well metadata from the Norwegian borehole database to Fimbul.jl
geothermal simulation input parameters. Provides case type selection based
on well type, sensible defaults for the Oslo region, parameter validation,
and simulation orchestration.

Supported Fimbul.jl case types:
- AGS  (Advanced Geothermal System) — closed-loop single energy well
- BTES (Borehole Thermal Energy Storage) — well park arrays

Typical workflow:
1. `well_to_simulation_params(properties)` — convert metadata to a parameter Dict
2. Edit parameters in the frontend
3. `run_fimbul_simulation(params)` — execute via Fimbul.jl
"""

using Fimbul
using Dates
using Jutul
using JutulDarcy
import Base64: base64encode
# CairoMakie is loaded separately (see _start_cairo_warmup in capability.py)
# so that including this file doesn't block the Julia kernel during session start.

# ── Server-side state for lazy reservoir image rendering ─────────────────────

const _sim_case = Ref{Any}(nothing)
const _sim_states = Ref{Any}(nothing)
const _sim_state0 = Ref{Any}(nothing)
const _image_cache = Dict{String, String}()
const _colorrange_cache = Dict{String, Tuple{Float64, Float64}}()
const _render_lock = ReentrantLock()

"""Print a timestamped progress line.

The caller (a Julia kernel eval with `on_chunk` streaming) picks these up live
from stdout, the same way `run_julia` streams any other long computation's
output — there is no separate log to poll here."""
function _sim_log_push!(msg::AbstractString)
    println("[$(Dates.format(now(), "HH:mm:ss"))] $msg")
end

# ── Physical constants for parameter estimation ──────────────────────────────

const WATER_VOLUMETRIC_HEAT_CAPACITY = 4.18e6  # ρ·cp of water [J/(m³·K)]
const ASSUMED_DELTA_T = 5.0                     # assumed temperature difference [K]
const SECONDS_PER_HOUR = 3600.0

# ── Case type enum ───────────────────────────────────────────────────────────

"""Supported Fimbul.jl simulation case types."""
@enum SimCaseType SIM_AGS SIM_BTES

const CASE_TYPE_LABELS = Dict(
    SIM_AGS  => "Advanced Geothermal System (AGS)",
    SIM_BTES => "Borehole Thermal Energy Storage (BTES)",
)

const CASE_TYPE_DESCRIPTIONS = Dict(
    SIM_AGS  => "Closed-loop heat exchanger in a single deep borehole. Suitable for individual energy wells.",
    SIM_BTES => "Array of closely-spaced boreholes for seasonal heat storage. Suitable for well parks.",
)

# ── Layer → Case type mapping ────────────────────────────────────────────────

"""Layer names for which simulation is supported."""
const SIMULATABLE_LAYERS = Dict(
    "EnergiBrønn" => SIM_AGS,
    "BrønnPark"   => SIM_BTES,
)

"""
    select_case_type(layer_name) -> Union{SimCaseType, Nothing}

Choose the most appropriate Fimbul simulation case type based on the
borehole layer (well type) from the Norwegian database.

Returns `nothing` for well types that are not simulatable.
"""
function select_case_type(layer_name::AbstractString)
    return get(SIMULATABLE_LAYERS, layer_name, nothing)
end

"""
    is_simulatable(layer_name) -> Bool

Return `true` if the given layer name supports simulation.
"""
function is_simulatable(layer_name::AbstractString)
    return haskey(SIMULATABLE_LAYERS, layer_name)
end

# ── Oslo-region defaults ─────────────────────────────────────────────────────

"""Sensible default parameters for the Oslo geological region."""
const OSLO_DEFAULTS = Dict{String,Any}(
    "surface_temperature"       => 7.0,     # °C — Oslo annual mean
    "geothermal_gradient"       => 0.025,   # K/m — typical Norwegian basement rock
    "rock_thermal_conductivity" => 3.0,     # W/(m·K) — Oslo gneiss/granite
    "rock_heat_capacity"        => 850.0,   # J/(kg·K)
    "rock_density"              => 2650.0,  # kg/m³ — crystalline rock
    "porosity"                  => 0.01,    # fractional — tight crystalline rock
    "permeability"              => 0.1,     # mD — low-permeability basement
)

# ── Parameter metadata ───────────────────────────────────────────────────────

"""
Metadata describing each simulation parameter.
Keys: label, unit, min, max, step, tooltip, group.
"""
const PARAM_METADATA = Dict{String,Dict{String,Any}}(
    # Well / geometry
    "well_depth"                => Dict{String,Any}("label" => "Well depth",                "unit" => "m",       "min" => 10,    "max" => 8000,  "step" => 10,    "tooltip" => "Total drilled depth of the well",                           "group" => "Well Geometry"),
    "borehole_diameter"         => Dict{String,Any}("label" => "Borehole diameter",         "unit" => "mm",      "min" => 50,    "max" => 500,   "step" => 1,     "tooltip" => "Diameter of the borehole",                                   "group" => "Well Geometry"),
    # Rock properties
    "surface_temperature"       => Dict{String,Any}("label" => "Surface temperature",       "unit" => "°C",      "min" => -10,   "max" => 40,    "step" => 0.5,   "tooltip" => "Mean annual temperature at the surface",                     "group" => "Rock Properties"),
    "geothermal_gradient"       => Dict{String,Any}("label" => "Geothermal gradient",       "unit" => "K/m",     "min" => 0.01,  "max" => 0.10,  "step" => 0.005, "tooltip" => "Rate of temperature increase with depth",                    "group" => "Rock Properties"),
    "rock_thermal_conductivity" => Dict{String,Any}("label" => "Thermal conductivity",      "unit" => "W/(m·K)", "min" => 0.5,   "max" => 10.0,  "step" => 0.1,   "tooltip" => "Thermal conductivity of the surrounding rock",               "group" => "Rock Properties"),
    "rock_heat_capacity"        => Dict{String,Any}("label" => "Rock heat capacity",        "unit" => "J/(kg·K)","min" => 100,   "max" => 2000,  "step" => 50,    "tooltip" => "Specific heat capacity of the rock matrix",                  "group" => "Rock Properties"),
    "porosity"                  => Dict{String,Any}("label" => "Porosity",                  "unit" => "–",       "min" => 0.001, "max" => 0.5,   "step" => 0.01,  "tooltip" => "Rock porosity (volume fraction)",                            "group" => "Rock Properties"),
    "permeability"              => Dict{String,Any}("label" => "Permeability",              "unit" => "mD",      "min" => 0.001, "max" => 5000,  "step" => 1,     "tooltip" => "Rock permeability in millidarcys",                           "group" => "Rock Properties"),
    # Grid
    "num_segments"              => Dict{String,Any}("label" => "Well segments",             "unit" => "–",       "min" => 2,     "max" => 100,   "step" => 1,     "tooltip" => "Number of vertical grid cells in the 3D well visualisation", "group" => "Visualisation"),
    # Operation
    "temperature_inj"           => Dict{String,Any}("label" => "Injection temperature",     "unit" => "°C",      "min" => 5,     "max" => 100,   "step" => 1,     "tooltip" => "Temperature of injected fluid",                              "group" => "Operation"),
    "flow_rate"                 => Dict{String,Any}("label" => "Flow rate",                 "unit" => "m³/h",    "min" => 1,     "max" => 500,   "step" => 1,     "tooltip" => "Volumetric flow rate",                                       "group" => "Operation"),
    "num_years"                 => Dict{String,Any}("label" => "Simulation years",          "unit" => "yr",      "min" => 1,     "max" => 500,   "step" => 1,     "tooltip" => "Duration of the simulation",                                 "group" => "Simulation"),
    # BTES-specific
    "num_wells_btes"            => Dict{String,Any}("label" => "Number of wells",           "unit" => "–",       "min" => 4,     "max" => 200,   "step" => 1,     "tooltip" => "Total number of boreholes in the BTES array",                "group" => "BTES Layout"),
    "num_sectors"               => Dict{String,Any}("label" => "Number of sectors",         "unit" => "–",       "min" => 1,     "max" => 20,    "step" => 1,     "tooltip" => "Number of sectors the wells are divided into",               "group" => "BTES Layout"),
    "well_spacing"              => Dict{String,Any}("label" => "Well spacing",              "unit" => "m",       "min" => 2,     "max" => 20,    "step" => 0.5,   "tooltip" => "Horizontal spacing between adjacent boreholes",              "group" => "BTES Layout"),
    "temperature_charge"        => Dict{String,Any}("label" => "Charge temperature",        "unit" => "°C",      "min" => 30,    "max" => 150,   "step" => 1,     "tooltip" => "Injection temperature during charging",                      "group" => "Operation"),
    "temperature_discharge"     => Dict{String,Any}("label" => "Discharge temperature",     "unit" => "°C",      "min" => 5,     "max" => 50,    "step" => 1,     "tooltip" => "Injection temperature during discharging",                   "group" => "Operation"),
    "rate_charge"               => Dict{String,Any}("label" => "Charge rate",               "unit" => "L/s",     "min" => 0.1,   "max" => 50,    "step" => 0.1,   "tooltip" => "Injection rate during charging per sector",                   "group" => "Operation"),
)

"""Parameter keys used for each case type, in display order."""
const CASE_PARAMS = Dict{SimCaseType, Vector{String}}(
    SIM_AGS => [
        "well_depth", "borehole_diameter",
        "surface_temperature", "geothermal_gradient",
        "rock_thermal_conductivity", "rock_heat_capacity",
        "porosity", "permeability",
        "temperature_inj", "flow_rate", "num_segments", "num_years",
    ],
    SIM_BTES => [
        "well_depth", "num_wells_btes", "num_sectors", "well_spacing",
        "surface_temperature", "geothermal_gradient",
        "rock_thermal_conductivity", "rock_heat_capacity",
        "temperature_charge", "temperature_discharge",
        "rate_charge", "num_segments", "num_years",
    ],
)

"""Default values for all parameters."""
const PARAM_DEFAULTS = Dict{String,Any}(
    "well_depth"                => 200.0,
    "borehole_diameter"         => 140.0,
    "temperature_inj"           => 25.0,
    "flow_rate"                 => 25.0,
    "num_years"                 => 25,
    "num_segments"              => 10,
    "num_wells_btes"            => 48,
    "num_sectors"               => 6,
    "well_spacing"              => 5.0,
    "temperature_charge"        => 90.0,
    "temperature_discharge"     => 10.0,
    "rate_charge"               => 0.5,
)

# ── Well metadata → simulation parameters ────────────────────────────────────

"""
    well_to_simulation_params(properties::Dict) -> Dict{String,Any}

Convert well metadata (GeoJSON properties) from the Norwegian borehole
database into Fimbul.jl simulation parameters.  Properties that are not
available in the dataset are filled with sensible Oslo-region defaults.

Returns a Dict with keys:
- `"simulatable"` — whether this well type supports simulation
- `"case_type"` — string label ("AGS", "BTES") or `nothing`
- `"case_label"` — human-readable case name
- `"case_description"` — short description
- `"well_id"` — well identifier from the metadata
- `"parameters"` — Dict of simulation parameter name → value
- `"parameter_order"` — ordered list of parameter keys for this case
- `"metadata"` — per-parameter metadata (label, unit, min, max, …)
- `"sources"` — Dict indicating whether each value came from "data" or "default"
"""
function well_to_simulation_params(properties::AbstractDict)
    layer_name = get(properties, "layer", "EnergiBrønn")
    layer_name = layer_name isa Nothing ? "EnergiBrønn" : string(layer_name)
    case_type = select_case_type(layer_name)
    well_id = _well_identifier(properties)

    if case_type === nothing
        return Dict{String,Any}(
            "simulatable"      => false,
            "case_type"        => nothing,
            "well_id"          => well_id,
            "parameters"       => Dict{String,Any}(),
            "parameter_order"  => String[],
            "metadata"         => Dict{String,Any}(),
            "sources"          => Dict{String,String}(),
        )
    end

    case_key = case_type == SIM_AGS ? "AGS" : "BTES"

    # Extract known values from metadata
    depth = _numeric(get(properties, "boretLengde", nothing))
    diameter = _numeric(get(properties, "diameterBorehull", nothing))
    n_energy_wells = _numeric(get(properties, "antallEnergiBrønner", nothing))
    heating_power  = _numeric(get(properties, "brønnpVEffekt", nothing))

    # Build parameter dict with source tracking
    params  = Dict{String,Any}()
    sources = Dict{String,String}()
    param_keys = CASE_PARAMS[case_type]

    for key in param_keys
        val, src = _resolve_param(key, depth, diameter, n_energy_wells, heating_power)
        params[key]  = val
        sources[key] = src
    end

    # Collect metadata for the parameters used
    meta = Dict{String,Any}()
    for key in param_keys
        if haskey(PARAM_METADATA, key)
            meta[key] = PARAM_METADATA[key]
        end
    end

    return Dict{String,Any}(
        "simulatable"      => true,
        "case_type"        => case_key,
        "case_label"       => CASE_TYPE_LABELS[case_type],
        "case_description" => CASE_TYPE_DESCRIPTIONS[case_type],
        "well_id"          => well_id,
        "parameters"       => params,
        "parameter_order"  => param_keys,
        "metadata"         => meta,
        "sources"          => sources,
    )
end

# ── Parameter resolution helpers ─────────────────────────────────────────────

"""Resolve a single parameter from well metadata or defaults."""
function _resolve_param(key, depth, diameter, n_energy_wells, heating_power)
    # Well geometry — from data when available
    if key == "well_depth" && depth !== nothing
        return (depth, "data")
    end
    if key == "borehole_diameter" && diameter !== nothing
        return (diameter, "data")
    end
    # BTES — derive from well park metadata
    if key == "num_wells_btes" && n_energy_wells !== nothing
        return (max(4, round(Int, n_energy_wells)), "data")
    end
    if key == "flow_rate" && heating_power !== nothing
        rate_m3h = heating_power / (WATER_VOLUMETRIC_HEAT_CAPACITY * ASSUMED_DELTA_T) * SECONDS_PER_HOUR
        return (max(1.0, round(rate_m3h; digits=1)), "data")
    end

    # Oslo-region defaults
    if haskey(OSLO_DEFAULTS, key)
        return (OSLO_DEFAULTS[key], "default")
    end

    # General defaults
    val = get(PARAM_DEFAULTS, key, 0.0)
    return (val, "default")
end

"""Convert a value to Float64 if possible, else return nothing."""
function _numeric(val)
    val === nothing && return nothing
    val isa Number  && return Float64(val)
    try
        return parse(Float64, string(val))
    catch
        return nothing
    end
end

"""Build a human-readable well identifier from metadata."""
function _well_identifier(props)
    nr = get(props, "brønnNr", nothing)
    nr !== nothing && return "Well #$(nr)"
    pk = get(props, "brønnParkNr", nothing)
    pk !== nothing && return "Well Park #$(pk)"
    return "Selected Well"
end

# ── Validation ───────────────────────────────────────────────────────────────

"""
    validate_simulation_params(params::Dict) -> Vector{Tuple{String,String}}

Validate simulation parameters against their metadata ranges.
Returns a list of `(param_key, error_message)` tuples; empty if valid.
"""
function validate_simulation_params(params::AbstractDict)
    errors = Tuple{String,String}[]
    case_type = get(params, "case_type", nothing)
    parameters = get(params, "parameters", params)

    for (key, val) in parameters
        meta = get(PARAM_METADATA, key, nothing)
        meta === nothing && continue
        v = _numeric(val)
        v === nothing && continue
        mn = get(meta, "min", -Inf)
        mx = get(meta, "max",  Inf)
        if v < mn
            push!(errors, (key, "$(meta["label"]) must be ≥ $mn $(meta["unit"])"))
        elseif v > mx
            push!(errors, (key, "$(meta["label"]) must be ≤ $mx $(meta["unit"])"))
        end
    end

    # Fimbul.btes divides wells evenly into num_sectors and special-cases the
    # innermost/outermost well of each sector — a sector left with exactly one
    # well hits both special cases at once and Fimbul indexes past the end of a
    # 1-element array. Catch it here with a clear message instead of a bare
    # BoundsError from inside the simulator.
    if case_type == "BTES"
        n = _numeric(get(parameters, "num_wells_btes", nothing))
        s = _numeric(get(parameters, "num_sectors", nothing))
        if n !== nothing && s !== nothing && s > 0 && n < 2 * s
            push!(errors, ("num_wells_btes",
                "Number of wells must be at least 2x the number of sectors " *
                "(need $(round(Int, 2 * s)) wells for $(round(Int, s)) sectors, " *
                "or fewer sectors) — each sector needs at least 2 wells"))
        end
    end

    return errors
end

# ── Simulation runner ────────────────────────────────────────────────────────

"""
    run_fimbul_simulation(setup::Dict) -> Dict{String,Any}

Execute a Fimbul.jl simulation using the prepared parameter set.

Uses the Fimbul.jl package to run the actual simulation. Falls back to
mock results only when `mock=true` is passed (for testing).
"""
function run_fimbul_simulation(setup::AbstractDict; mock::Bool=false)
    case_type = get(setup, "case_type", "AGS")
    params    = setup["parameters"]

    if mock
        return _run_mock_simulation(case_type, params)
    end

    return _run_fimbul_live(case_type, params)
end

"""Run simulation using Fimbul.jl."""
function _run_fimbul_live(case_type, params)
    try
        _sim_log_push!("Initializing $case_type simulation...")

        # Note: num_segments controls the 3D wellbore visualisation only
        # and is not passed to the simulator.
        if case_type == "AGS"
            _sim_log_push!("Creating AGS case with well depth=$(get(params, "well_depth", "?"))m...")
            case = Fimbul.ags(;
                porosity                  = params["porosity"],
                permeability              = params["permeability"] * 1e-3 * Fimbul.darcy,
                rock_thermal_conductivity = params["rock_thermal_conductivity"] * Fimbul.watt / (Fimbul.meter * Fimbul.Kelvin),
                rock_heat_capacity        = params["rock_heat_capacity"] * Fimbul.joule / (Fimbul.kilogram * Fimbul.Kelvin),
                temperature_surface       = Fimbul.convert_to_si(params["surface_temperature"], :Celsius),
                thermal_gradient          = params["geothermal_gradient"] * Fimbul.Kelvin / Fimbul.meter,
                rate                      = params["flow_rate"] * Fimbul.meter^3 / Fimbul.hour,
                temperature_inj           = Fimbul.convert_to_si(params["temperature_inj"], :Celsius),
                num_years                 = round(Int, params["num_years"]),
            )
        elseif case_type == "BTES"
            _sim_log_push!("Creating BTES case with $(get(params, "num_wells_btes", "?")) wells...")
            case = Fimbul.btes(;
                num_wells            = round(Int, params["num_wells_btes"]),
                num_sectors          = round(Int, params["num_sectors"]),
                well_spacing         = params["well_spacing"],
                temperature_charge   = Fimbul.convert_to_si(params["temperature_charge"], :Celsius),
                temperature_discharge = Fimbul.convert_to_si(params["temperature_discharge"], :Celsius),
                rate_charge          = params["rate_charge"] * Fimbul.litre / Fimbul.second,
                temperature_surface  = Fimbul.convert_to_si(params["surface_temperature"], :Celsius),
                geothermal_gradient  = params["geothermal_gradient"] * Fimbul.Kelvin / Fimbul.meter,
                num_years            = round(Int, params["num_years"]),
            )
        else
            return Dict("status" => "error", "message" => "Unknown case type: $case_type")
        end

        _sim_log_push!("Case created. Starting reservoir simulation...")
        _sim_log_push!("This may take several minutes depending on model size.")

        # Fimbul's own progress output (progress bars etc.) goes straight to
        # stdout; the caller's kernel eval streams it live, same as any other
        # long-running Julia computation — no manual capture needed here.
        results = Fimbul.simulate_reservoir(case)

        _sim_log_push!("Simulation completed. Extracting results...")

        # Store case and states for lazy reservoir image rendering
        _sim_case[] = case
        _sim_states[] = results.states
        _sim_state0[] = get(case.state0, :Reservoir, nothing)
        empty!(_image_cache)
        empty!(_colorrange_cache)

        # Extract well data from results with unit conversion
        well_data = Dict{String,Any}()
        for (wname, wdata) in pairs(results.wells)
            wdict = Dict{String,Any}()
            for (k, v) in pairs(wdata)
                if v isa AbstractVector && eltype(v) <: Number
                    ckey, cvals = _convert_well_variable(string(k), collect(Float64, v))
                    wdict[ckey] = cvals
                end
            end
            well_data[string(wname)] = wdict
        end

        # Convert timestamps from seconds to days
        timestamps = collect(Float64, Fimbul.convert_from_si.(results.time, :day))

        # Extract reservoir state variable names and count
        reservoir_vars = String[]
        if !isempty(results.states)
            for (k, v) in pairs(results.states[1])
                if v isa AbstractVector{<:Real}
                    push!(reservoir_vars, string(k))
                end
            end
        end

        _sim_log_push!("Results extracted: $(length(well_data)) well(s), $(length(results.states)) timesteps.")
        _sim_log_push!("Reservoir variables: $(join(reservoir_vars, ", "))")
        _sim_log_push!("Simulation finished successfully.")

        return Dict{String,Any}(
            "status"         => "completed",
            "message"        => "Simulation completed successfully.",
            "well_data"      => well_data,
            "timestamps"     => timestamps,
            "num_steps"      => length(results.states),
            "reservoir_vars" => reservoir_vars,
        )
    catch e
        _sim_log_push!("ERROR: $(sprint(showerror, e))")
        return Dict{String,Any}(
            "status"  => "error",
            "message" => "Simulation failed: $(sprint(showerror, e))",
        )
    end
end

# ── Unit conversion helpers (following FimbulApp.jl pattern) ──────────────────

const _K_to_C = 273.15
const _m3s_to_Ls = 1000.0
const _Pa_to_bar = 1e-5

"""Convert a well output variable from SI units to user-friendly units."""
function _convert_well_variable(name::String, values::Vector{Float64})
    ln = lowercase(name)
    if occursin("temperature", ln)
        return name * " [°C]", values .- _K_to_C
    elseif occursin("rate", ln) && !occursin("mass", ln)
        return name * " [L/s]", values .* _m3s_to_Ls
    elseif occursin("pressure", ln) || ln == "bhp"
        return name * " [bar]", values .* _Pa_to_bar
    else
        return name, values
    end
end

# ── Mock simulation (for testing only) ───────────────────────────────────────

const MOCK_TEMP_DECAY_FACTOR = 0.3
const MOCK_SEASONAL_AMPLITUDE = 0.5
const MOCK_BTES_WARMUP_FRACTION = 4
const DAYS_PER_YEAR = 365.25

"""Generate mock simulation results for testing."""
function _run_mock_simulation(case_type, params)
    _sim_log_push!("Starting mock $case_type simulation...")

    n_years = round(Int, get(params, "num_years", 25))
    n_steps = n_years * 12
    dt = range(0, n_years * DAYS_PER_YEAR; length=n_steps)
    timestamps = collect(Float64, dt)

    depth = get(params, "well_depth", 200.0)
    T_surface = get(params, "surface_temperature", 7.0)
    gradient = get(params, "geothermal_gradient", 0.025)
    T_bottom = T_surface + gradient * depth

    _sim_log_push!("Generating well data ($n_steps timesteps)...")

    T_prod = [T_bottom - MOCK_TEMP_DECAY_FACTOR * log(1 + t / DAYS_PER_YEAR) + MOCK_SEASONAL_AMPLITUDE * sin(2π * t / DAYS_PER_YEAR) for t in timestamps]
    T_inj  = fill(get(params, "temperature_inj", 25.0), n_steps)
    rate   = fill(get(params, "flow_rate", 25.0), n_steps)

    well_data = Dict{String,Any}()
    if case_type == "AGS"
        well_data["Producer"] = Dict{String,Any}(
            "temperature" => T_prod,
            "mass_rate"   => rate,
        )
    else  # BTES
        T_out = [T_surface + (get(params, "temperature_charge", 90.0) - T_surface) * (1 - exp(-i / (n_steps / MOCK_BTES_WARMUP_FRACTION))) * (0.5 + MOCK_SEASONAL_AMPLITUDE * sin(2π * t / DAYS_PER_YEAR)) for (i, t) in enumerate(timestamps)]
        well_data["BTES Array"] = Dict{String,Any}(
            "temperature" => T_out,
            "mass_rate"   => fill(get(params, "rate_charge", 0.5), n_steps),
        )
    end

    # Generate mock reservoir vars (names only, no full grid data for mock)
    reservoir_vars = ["Pressure", "Temperature"]

    _sim_log_push!("Mock simulation completed.")

    return Dict{String,Any}(
        "status"         => "completed",
        "message"        => "Mock simulation completed (for testing).",
        "well_data"      => well_data,
        "timestamps"     => timestamps,
        "num_steps"      => n_steps,
        "reservoir_vars" => reservoir_vars,
    )
end

# ── Reservoir image rendering (following FimbulApp.jl pattern) ────────────────

"""
    _convert_reservoir_variable(var, values; delta=false)

Convert reservoir state variable from SI units to user-friendly units.
Returns `(converted_values, unit_label)`.
"""
function _convert_reservoir_variable(var::AbstractString, values; delta::Bool=false)
    ln = lowercase(var)
    if occursin("temperature", ln)
        if delta
            return values, "°C"  # K delta = °C delta
        else
            return values .- _K_to_C, "°C"
        end
    elseif occursin("pressure", ln)
        return values .* _Pa_to_bar, "bar"
    else
        return values, ""
    end
end

"""
    _get_colorrange(var, delta)

Compute global min/max color range across all stored timesteps for consistent coloring.
Results are cached for performance.
"""
function _get_colorrange(var::AbstractString, delta::Bool)
    cache_key = "$var:$delta"
    haskey(_colorrange_cache, cache_key) && return _colorrange_cache[cache_key]

    states = _sim_states[]
    state0 = _sim_state0[]
    isnothing(states) && return (0.0, 1.0)

    sym = Symbol(var)
    global_min = Inf
    global_max = -Inf
    for i in 1:length(states)
        raw = if delta && !isnothing(state0) && haskey(state0, sym)
            states[i][sym] .- state0[sym]
        else
            states[i][sym]
        end
        vals, _ = _convert_reservoir_variable(var, raw; delta=delta)
        lo, hi = extrema(vals)
        global_min = min(global_min, lo)
        global_max = max(global_max, hi)
    end

    result = (global_min, global_max)
    _colorrange_cache[cache_key] = result
    return result
end

"""
    render_reservoir_image(var, step; delta=false)

Render a reservoir state image as a base64-encoded PNG string.
Uses CairoMakie + Jutul.plot_cell_data! for 3D visualization.
Results are cached server-side for performance.

# Arguments
- `var::AbstractString`: Variable name (e.g. "Temperature", "Pressure")
- `step::Int`: 1-based timestep index
- `delta::Bool`: If true, show difference from initial state
"""
function render_reservoir_image(var::AbstractString, step::Int; delta::Bool=false)
    cache_key = "$var:$step:$delta"
    haskey(_image_cache, cache_key) && return _image_cache[cache_key]

    case = _sim_case[]
    states = _sim_states[]
    (isnothing(case) || isnothing(states)) && return ""
    (step < 1 || step > length(states)) && return ""

    try
        res_model = Fimbul.reservoir_model(case.model)
        mesh = Fimbul.physical_representation(res_model.data_domain)
        state = states[step]
        sym = Symbol(var)
        title = "$var at step $step"
        if delta
            state0 = _sim_state0[]
            (isnothing(state0) || !haskey(state0, sym)) && return ""
            raw_vals = state[sym] .- state0[sym]
            title = "Δ $var at step $step"
        else
            raw_vals = state[sym]
        end
        plot_vals, unit_label = _convert_reservoir_variable(var, raw_vals; delta=delta)
        if !isempty(unit_label)
            title *= " [$unit_label]"
        end
        cmin, cmax = _get_colorrange(var, delta)
        img = lock(_render_lock) do
            fig = Figure(size = (800, 600))
            ax = Axis3(fig[1, 1], title = title, aspect = :data, zreversed = true)
            p = Jutul.plot_cell_data!(ax, mesh, plot_vals, outer=true,
                colormap=:seaborn_icefire_gradient, colorrange=(cmin, cmax))
            Colorbar(fig[1, 2], p)
            io = IOBuffer()
            show(io, MIME("image/png"), fig)
            base64encode(take!(io))
        end
        _image_cache[cache_key] = img
        return img
    catch e
        err_msg = sprint(showerror, e; context=:limit => true)
        if length(err_msg) > 500
            err_msg = err_msg[1:500] * "…"
        end
        @warn "Failed to render reservoir image for $var step $step (delta=$delta): $err_msg"
        return ""
    end
end

# ── Fimbul validation (prescribed T_in from pygfunction) ──────────────────────

"""
    run_btes_validation(setup) -> Dict{String,Any}

Run a Fimbul BTES simulation with a prescribed per-period injection temperature
timeseries (T_in) derived from a prior pygfunction run. Returns T_out and
extracted heat energy per period.

Expected keys in `setup`:
- `"num_boreholes"`, `"num_sectors"` — borehole field layout
- `"H"`, `"B"` — active depth [m] and well spacing [m]
- `"k_s"` — ground thermal conductivity [W/(m·K)]
- `"T_surface_C"` — mean annual surface temperature [°C]
- `"G"` — geothermal gradient [K/m]
- `"m_flow_per_borehole"` — mass flow rate per sector [kg/s]
- `"T_in_C"` — prescribed injection temperatures, one per period [°C]
- `"durations_s"` — duration of each period [s]
"""
function run_btes_validation(setup::AbstractDict)
    try
        f64(k)      = Float64(setup[k])
        num_wells   = round(Int, setup["num_boreholes"])
        num_sectors = round(Int, setup["num_sectors"])
        H, B, k_s   = f64("H"), f64("B"), f64("k_s")
        T_surf_C, G = f64("T_surface_C"), f64("G")
        m_flow, rho_f, cp_f = f64("m_flow_per_borehole"), f64("rho_fluid"), f64("cp_fluid")
        T_in_C      = Float64.(setup["T_in_C"])
        durations_s = Float64.(setup["durations_s"])
        n_periods   = length(T_in_C)
        @assert length(durations_s) == n_periods "T_in_C and durations_s must have equal length"

        T_surf_K    = Fimbul.convert_to_si(T_surf_C, :Celsius)
        G_SI        = G * Fimbul.Kelvin/Fimbul.meter
        rate_SI     = m_flow / rho_f
        T_in_mean_K = Fimbul.convert_to_si(sum(T_in_C) / n_periods, :Celsius)
        depths      = [0.0, 0.5, H, H + 15.0]

        _sim_log_push!("Building BTES model: $num_wells wells, $num_sectors sectors, H=$(H) m, B=$(B) m...")

        # Call btes() with a 1-year dummy schedule purely to obtain the model,
        # state0, and sectors dict. The schedule is discarded immediately.
        dummy = btes(
            num_wells             = num_wells,
            num_sectors           = num_sectors,
            well_spacing          = B,
            depths                = depths,
            thermal_conductivity  = [0.034, k_s, k_s] .* Fimbul.watt/Fimbul.meter/Fimbul.Kelvin,
            geothermal_gradient   = G_SI,
            temperature_charge    = T_in_mean_K,
            temperature_discharge = T_in_mean_K,
            rate_charge           = rate_SI,
            temperature_surface   = T_surf_K,
            num_years             = 1,
        )
        model, state0, sectors = dummy.model, dummy.state0, dummy.input_data[:sectors]

        # Reconstruct boundary conditions matching btes.jl: top + sides, not bottom
        bc, _, _ = Fimbul.set_dirichlet_bcs(
            model, [:top, :sides];
            temperature_surface = T_surf_K,
            geothermal_gradient = G_SI,
            output_state        = false,
        )

        rho      = reservoir_model(model).system.rho_ref[1]
        rate_tgt = TotalRateTarget(rate_SI)
        bhp_tgt  = BottomHolePressureTarget(1.0Jutul.si_unit(:atm))
        is_sup(w) = endswith(String(w), "_supply")
        is_ret(w) = endswith(String(w), "_return")

        # Build one force set per period — only injection temperature varies
        forces_list = Any[]
        for T_C in T_in_C
            T_K  = Fimbul.convert_to_si(T_C, :Celsius)
            ctrl = Dict()
            for (_, sec_wells) in pairs(sectors)
                supply = filter(is_sup, collect(sec_wells))
                retrn  = filter(is_ret, collect(sec_wells))
                for (k, w_sup) in enumerate(supply)
                    ctrl[w_sup] = k == 1 ?
                        InjectorControl(rate_tgt, [1.0], density=rho, temperature=T_K) :
                        InjectorControl(JutulDarcy.ReinjectionTarget([retrn[k-1]]), [1.0],
                            density=rho, temperature=NaN; check=false)
                    ctrl[retrn[k]] = ProducerControl(bhp_tgt)
                end
            end
            push!(forces_list, setup_reservoir_forces(model, control=ctrl, bc=bc))
        end

        # One report per period so results index 1-to-1 with T_in_C
        dt_vec, forces_vec, _ = make_schedule(forces_list, durations_s; num_reports=1)

        _sim_log_push!("Schedule: $n_periods periods, $(round(sum(durations_s)/86400; digits=1)) days total.")
        _sim_log_push!("Starting Fimbul validation simulation...")

        results = simulate_reservoir(JutulCase(model, dt_vec, forces_vec, state0=state0))

        _sim_log_push!("Simulation done. Extracting results...")

        # T_out = outlet of the last return well in each sector, averaged across sectors.
        # sectors[:Sk] = [supply_1, return_1, ..., supply_n, return_n], so [end] is the sector outlet.
        T_out_K = sum(
            collect(Float64, results.wells[filter(is_ret, collect(sw))[end]][:temperature])
            for (_, sw) in pairs(sectors)
        ) ./ length(sectors)
        T_out_C = Fimbul.convert_from_si.(T_out_K, :Celsius)

        # Extracted energy per period [kWh] = N_sectors · ṁ · cp · ΔT · Δt
        energy_kWh = [
            num_sectors * m_flow * cp_f * (T_out_C[i] - T_in_C[i]) * durations_s[i] / 3.6e6
            for i in 1:n_periods
        ]

        total_kWh = sum(energy_kWh)
        _sim_log_push!("Total extracted energy: $(round(total_kWh; digits=1)) kWh.")

        return Dict{String,Any}(
            "status"           => "completed",
            "T_in_C"           => collect(T_in_C),
            "T_out_C"          => collect(T_out_C),
            "energy_kWh"       => collect(energy_kWh),
            "durations_s"      => collect(durations_s),
            "total_energy_kWh" => total_kWh,
            "n_periods"        => n_periods,
        )
    catch e
        bt  = catch_backtrace()
        msg = sprint(showerror, e, bt)
        _sim_log_push!("ERROR: $msg")
        return Dict{String,Any}(
            "status"  => "error",
            "message" => "Validation failed: $msg",
        )
    end
end
