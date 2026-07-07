"""
Julia overlay for pygfunction_sim.py using PythonCall.jl.

Calls simulate_borehole_temperatures (and related helpers) directly in the
same Python interpreter that runs jutul-agent, with no subprocess or JSON
serialisation round-trip. PythonCall.jl is used for the bridge.

Pre-requisite: before include()ing this file the calling template must set
  ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
  ENV["JULIA_PYTHONCALL_EXE"]  = "<path to the project venv python>"
so that PythonCall links against the venv (which has pygfunction installed)
rather than creating its own Conda environment.

Typical usage from a Julia template (in capability.py):
  ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
  ENV["JULIA_PYTHONCALL_EXE"]  = raw"<PYTHON_EXE>"
  if !isdefined(Main, :run_borehole_simulation)
      include(raw"<THIS FILE>")
  end
  result = run_borehole_simulation(timestamps, demand_kw; lat=..., lon=...)
  img_b64 = plot_borehole_temperatures(result; title="...")
"""

using PythonCall
using Base64
import CairoMakie

# Add the energy_demand_app directory to Python sys.path so pygfunction_sim
# is importable when the include() resolves relative to this julia/ sub-dir.
let
    pysys   = pyimport("sys")
    app_dir = normpath(joinpath(@__DIR__, ".."))
    already = any(pyconvert(String, p) == app_dir for p in pysys.path)
    already || pysys.path.insert(0, app_dir)
end

const _pygsim = pyimport("pygfunction_sim")
const _pd     = pyimport("pandas")

# Most-recent simulation result; re-used by plot_borehole_temperatures() so
# the caller does not have to pass it explicitly.
const _pygsim_result = Ref{Union{Dict{String,Any},Nothing}}(nothing)

"""
    run_borehole_simulation(timestamps, demand_kw; kwargs...) -> Dict{String,Any}

Run a pygfunction g-function borehole simulation in-process via PythonCall.

Parameters mirror `pygfunction_sim.simulate_borehole_temperatures`. When
`lat` and `lon` are provided, T_g is fetched from ERA5-Land (1991-2020
climate normal) instead of using the `T_g` default.

# Positional arguments
- `timestamps`: hourly timestamps (strings or any type supporting `string()`).
- `demand_kw`: hourly building thermal demand [kW].

# Keyword arguments (borehole field)
- `lat`, `lon`: site coordinates [degrees]. When provided, T_g is fetched from ERA5-Land.
- `H=200.0`, `D=4.0`, `r_b=0.070`: depth [m], burial depth [m], borehole radius [m].
- `N_1=1`, `N_2=1`, `B=5.0`: grid dimensions and spacing [m].
- `k_s=3.0`, `alpha=1.33e-6`: ground conductivity [W/(m·K)], diffusivity [m²/s].
- `COP=3.5`: heat-pump COP.
- `geothermal_gradient=0.025`: [K/m].

# Returns
`Dict` with keys `timestamps`, `Q_building_kW`, `Q_borehole_kW`, `T_in`, `T_out`.
"""
function run_borehole_simulation(
    timestamps::AbstractVector,
    demand_kw::AbstractVector;
    lat               = nothing,
    lon               = nothing,
    H::Float64        = 200.0,
    D::Float64        = 4.0,
    r_b::Float64      = 0.070,
    N_1::Int          = 1,
    N_2::Int          = 1,
    B::Float64        = 5.0,
    k_s::Float64      = 3.0,
    alpha::Float64    = 1.33e-6,
    COP::Float64      = 3.5,
    geothermal_gradient::Float64 = 0.025,
)
    local _pytb = pyimport("traceback")

    function _py_error_message(e::PythonCall.PyException)
        try
            lines = pyconvert(Vector{String}, _pytb.format_exception(e.t, e.v, e.b))
            return join(lines)
        catch
            return pyconvert(String, pybuiltins.str(e.v))
        end
    end

    df_py = _pd.DataFrame(
        pydict(time=pylist(string.(timestamps)), kW=pylist(collect(Float64, demand_kw)))
    )

    borehole_py = try
        _pygsim.BoreholeFieldParams(
            H=H, D=D, r_b=r_b, N_1=N_1, N_2=N_2, B=B, k_s=k_s, alpha=alpha,
        )
    catch e
        e isa PythonCall.PyException && error("BoreholeFieldParams failed:\n$(_py_error_message(e))")
        rethrow()
    end

    result_df = try
        _pygsim.simulate_borehole_temperatures(
            df_py,
            lat               = isnothing(lat) ? pybuiltins.None : Py(lat),
            lon               = isnothing(lon) ? pybuiltins.None : Py(lon),
            borehole          = borehole_py,
            COP               = COP,
            geothermal_gradient = geothermal_gradient,
        )
    catch e
        e isa PythonCall.PyException && error("simulate_borehole_temperatures failed:\n$(_py_error_message(e))")
        rethrow()
    end

    out = Dict{String,Any}(
        "timestamps"    => pyconvert(Vector{String},  result_df["time"].astype(pybuiltins.str).tolist()),
        "Q_building_kW" => pyconvert(Vector{Float64}, result_df["Q_building_kW"].tolist()),
        "Q_borehole_kW" => pyconvert(Vector{Float64}, result_df["Q_borehole_kW"].tolist()),
        "T_in"          => pyconvert(Vector{Float64}, result_df["T_in"].tolist()),
        "T_out"         => pyconvert(Vector{Float64}, result_df["T_out"].tolist()),
    )
    _pygsim_result[] = out
    return out
end

"""
    plot_borehole_temperatures([result]; title) -> String

Render a 4-panel CairoMakie figure from a borehole simulation result and
return a base64-encoded PNG string.

Panels (top to bottom): building demand [kW], T_in [°C], T_out [°C],
T_out - T_in [°C]. The x-axis is the hour index; all four share the same
x range. When `result` is omitted, the most recent `run_borehole_simulation`
result stored in `_pygsim_result[]` is used.
"""
function plot_borehole_temperatures(
    result::Union{Dict{String,Any},Nothing} = _pygsim_result[];
    title::String = "Borehole temperature simulation",
) :: String
    result === nothing && error(
        "No simulation result available — call run_borehole_simulation first."
    )

    n     = length(result["timestamps"])
    idx   = 1:n
    Q     = result["Q_building_kW"]   :: Vector{Float64}
    T_in  = result["T_in"]            :: Vector{Float64}
    T_out = result["T_out"]           :: Vector{Float64}
    dT    = T_out .- T_in

    fig = CairoMakie.Figure(size = (1000, 800))
    ax1 = CairoMakie.Axis(fig[1, 1]; ylabel = "Demand [kW]",   title)
    ax2 = CairoMakie.Axis(fig[2, 1]; ylabel = "T_in [°C]")
    ax3 = CairoMakie.Axis(fig[3, 1]; ylabel = "T_out [°C]")
    ax4 = CairoMakie.Axis(fig[4, 1]; ylabel = "ΔT [°C]",       xlabel = "Hour")

    CairoMakie.lines!(ax1, idx, Q,     color = :darkorange)
    CairoMakie.lines!(ax2, idx, T_in,  color = :steelblue)
    CairoMakie.lines!(ax3, idx, T_out, color = :seagreen)
    CairoMakie.lines!(ax4, idx, dT,    color = :mediumpurple)

    for ax in (ax1, ax2, ax3)
        CairoMakie.hidexdecorations!(ax; ticks = false, grid = false)
    end
    CairoMakie.linkxaxes!(ax1, ax2, ax3, ax4)

    io = IOBuffer()
    show(io, MIME("image/png"), fig)
    return base64encode(take!(io))
end
