from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pygfunction as gt


@dataclasses.dataclass
class BoreholeFieldParams:
    """Geometric and ground-thermal parameters for a borehole field."""

    H: float = 200.0  # active borehole depth [m] — geothermal-viz default
    D: float = 4.0  # buried depth from surface to top of active section [m]
    r_b: float = 0.070  # borehole radius [m] — from borehole_diameter 140 mm
    N_1: int = 1  # number of boreholes in x direction
    N_2: int = 1  # number of boreholes in y direction
    B: float = 5.0  # borehole spacing [m] — geothermal-viz default
    T_g: float = 7.0  # undisturbed ground temperature [degrees C] — Oslo mean
    k_s: float = 3.0  # ground thermal conductivity [W/(m K)] — Oslo gneiss/granite
    alpha: float = 1.33e-6  # ground thermal diffusivity [m^2/s] — k_s / (rho*cp) = 3.0 / (2650*850)


@dataclasses.dataclass
class PipeParams:
    """Single U-tube pipe and grout parameters."""

    r_in: float = 0.015  # inner pipe radius [m]
    r_out: float = 0.020  # outer pipe radius [m]
    D_s: float = 0.040  # shank spacing: borehole-centre to pipe-centre [m]
    k_p: float = 0.42  # pipe wall thermal conductivity [W/(m K)] -- HDPE
    k_g: float = 1.0  # grout thermal conductivity [W/(m K)]
    epsilon: float = 1e-6  # pipe inner-wall roughness [m]


@dataclasses.dataclass
class FluidParams:
    """Carrier fluid and flow parameters."""

    fluid_name: str = "Water"  # pygfunction fluid identifier
    concentration: float = 0.0  # antifreeze concentration [%] — 0 for pure water
    m_flow_per_borehole: float = 0.3  # mass flow rate per borehole [kg/s]


def simulate_borehole_temperatures(
    df_demand: pd.DataFrame,
    borehole: BoreholeFieldParams | None = None,
    pipe: PipeParams | None = None,
    fluid: FluidParams | None = None,
    COP: float = 3.5,
) -> pd.DataFrame:
    """Compute hourly fluid temperatures in a borehole field from a building thermal demand.

    Converts building heating demand to a ground extraction load via
    Q_borehole = Q_building * (COP - 1) / COP, then uses pygfunction's
    g-function / Claesson-Javed load aggregation / SingleUTube pipe model to
    obtain the borehole wall temperature T_b and carrier-fluid inlet and outlet
    temperatures T_in / T_out at each time step.

    Args:
        df_demand:
            DataFrame with a 'time' column and one kW demand column (any name).
            The first non-'time' column is used as the demand series.
        borehole:
            Borehole field geometry and ground thermal properties.
            Defaults to a single 200 m borehole in typical Norwegian ground.
        pipe:
            Single U-tube pipe and grout parameters.
            Defaults to 40 mm OD HDPE pipes in standard grout.
        fluid:
            Carrier fluid, antifreeze concentration, and mass flow rate.
            Defaults to 20 % mono-propylene glycol (MPG).
        COP:
            Heat-pump COP used to derive the ground extraction load.
            Q_borehole = Q_building * (COP - 1) / COP.

    Returns:
        DataFrame with columns:
            time            -- original timestamps
            Q_building_kW   -- building thermal demand [kW]
            Q_borehole_kW   -- ground extraction load [kW]
            T_in            -- fluid inlet temperature [degrees C]
            T_out           -- fluid outlet temperature [degrees C]
    """
    if borehole is None:
        borehole = BoreholeFieldParams()
    if pipe is None:
        pipe = PipeParams()
    if fluid is None:
        fluid = FluidParams()

    # -- demand series --
    df = df_demand.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    demand_cols = [c for c in df.columns if c != "time"]
    if not demand_cols:
        raise ValueError("df_demand must contain at least one column besides 'time'.")
    demand_col = demand_cols[0]

    Q_building_W = df[demand_col].to_numpy(dtype=float) * 1_000.0  # kW -> W
    n_hours = len(Q_building_W)
    dt = 3_600.0  # 1-hour time step [s]
    tmax = n_hours * dt

    # Ground extraction load: positive = heat drawn from the ground
    Q_borehole_W = Q_building_W * (COP - 1.0) / COP

    # -- borehole field --
    boreholes = gt.boreholes.rectangle_field(
        borehole.N_1,
        borehole.N_2,
        borehole.B,
        borehole.B,
        borehole.H,
        borehole.D,
        borehole.r_b,
    )
    N_b = len(boreholes)
    H_total = borehole.H * N_b  # total active borehole length [m]

    # -- fluid properties --
    fl = gt.media.Fluid(fluid.fluid_name, fluid.concentration)
    cp_f = fl.cp  # specific heat capacity [J/(kg K)]
    rho_f = fl.rho  # density [kg/m^3]
    mu_f = fl.mu  # dynamic viscosity [Pa s]
    k_f = fl.k  # thermal conductivity [W/(m K)]
    m_flow = fluid.m_flow_per_borehole

    # -- pipe thermal resistance: fluid film + pipe wall --
    R_p = gt.pipes.conduction_thermal_resistance_circular_pipe(pipe.r_in, pipe.r_out, pipe.k_p)
    h_f = gt.pipes.convective_heat_transfer_coefficient_circular_pipe(
        m_flow, pipe.r_in, mu_f, rho_f, k_f, cp_f, pipe.epsilon
    )
    R_fp = 1.0 / (h_f * 2.0 * np.pi * pipe.r_in) + R_p  # [(m K)/W]

    # -- SingleUTube pipe object (identical for every borehole in the field) --
    pos = [(-pipe.D_s, 0.0), (pipe.D_s, 0.0)]
    pipe_obj = gt.pipes.SingleUTube(
        pos,
        pipe.r_in,
        pipe.r_out,
        boreholes[0],
        borehole.k_s,
        pipe.k_g,
        R_fp,
    )

    # -- Claesson-Javed load aggregation --
    LoadAgg = gt.load_aggregation.ClaessonJaved(dt, tmax)
    time_req = LoadAgg.get_times_for_simulation()

    # -- g-function --
    gFunc = gt.gfunction.gFunction(
        boreholes,
        borehole.alpha,
        time=time_req,
        options={"nSegments": 8},
    )
    LoadAgg.initialize(gFunc.gFunc / (2.0 * np.pi * borehole.k_s))

    # -- hourly simulation loop --
    time_s = np.arange(1, n_hours + 1) * dt  # cumulative seconds
    T_in_arr = np.empty(n_hours)
    T_out_arr = np.empty(n_hours)

    for i in range(n_hours):
        LoadAgg.next_time_step(time_s[i])
        LoadAgg.set_current_load(Q_borehole_W[i] / H_total)  # W/m
        T_b = borehole.T_g - LoadAgg.temporal_superposition()

        Q_i = Q_borehole_W[i] / N_b  # W per borehole (parallel connection)
        T_in_arr[i] = pipe_obj.get_inlet_temperature(Q_i, T_b, m_flow, cp_f)
        T_out_arr[i] = pipe_obj.get_outlet_temperature(T_in_arr[i], T_b, m_flow, cp_f)

    return pd.DataFrame(
        {
            "time": df["time"].values,
            "Q_building_kW": Q_building_W / 1_000.0,
            "Q_borehole_kW": Q_borehole_W / 1_000.0,
            "T_in": T_in_arr,
            "T_out": T_out_arr,
        }
    )
