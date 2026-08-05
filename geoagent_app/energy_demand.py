from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from PROFet_API import run_profet_for_building

DemandType = Literal["DHW", "Space heating", "Total thermal heating"]


def get_energy_demand_timeseries(
    df_temperature: pd.DataFrame,
    selected_year: int,
    building_type_code: int | None = None,
    usable_floor_area_m2: float | None = None,
    efficiency_key: str = "Reg",
    demand_type: DemandType = "Total thermal heating",
    building_category: str | None = None,
) -> pd.DataFrame:
    """Return an hourly energy demand time series for the given building and year.

    Tries to obtain the result from the PROFet API. Falls back to a synthetic
    time series if the API call fails (e.g. no API key configured yet).

    Supply either ``building_type_code`` (Matrikkel code, normal WFS path) or
    ``building_category`` (PROFet category string, fallback when WFS is down).

    Args:
        df_temperature: DataFrame with columns 'time' and 'temperature_C'.
        selected_year: Calendar year to simulate.
        building_type_code: Matrikkel building type code.
        usable_floor_area_m2: Heated floor area [m²], or None to use a typical value.
        efficiency_key: Energy efficiency level ("Reg", "Eff-E", "Eff-N", "Vef").
        demand_type: Which demand component to return — "DHW", "Space heating",
            or "Total thermal heating".
        building_category: PROFet category (e.g. "House", "Office"). Used when
            building_type_code is None (WFS unavailable).

    Returns:
        DataFrame with columns 'time' and the demand in kW, one row per hour.
    """
    from PROFet_API import (
        TYPICAL_AREAS_M2,
        map_matrikkel_building_type_to_profet_category,
    )

    if building_type_code is not None:
        category = map_matrikkel_building_type_to_profet_category(building_type_code)
    elif building_category:
        category = building_category
    else:
        raise ValueError("Either building_type_code or building_category must be provided.")

    if usable_floor_area_m2 is None:
        usable_floor_area_m2 = TYPICAL_AREAS_M2[category]

    try:
        result = run_profet_for_building(
            df_temperature=df_temperature,
            selected_year=selected_year,
            building_category=category,
            usable_floor_area_m2=usable_floor_area_m2,
            efficiency_key=efficiency_key,
        )
        return _extract_timeseries(result, demand_type)
    except Exception as exc:
        print(f"[PROFet] API call failed ({exc}), falling back to synthetic demand.")
        return _make_synthetic_timeseries(
            df_temperature=df_temperature,
            demand_type=demand_type,
            building_category=category,
            efficiency_key=efficiency_key,
            usable_floor_area_m2=usable_floor_area_m2,
        )


def _extract_timeseries(
    profet_result: dict,
    demand_type: DemandType,
) -> pd.DataFrame:
    """Extract the requested demand component as a function of time from a PROFet API response.

    TODO: implement once the API response format is known.
    """
    raise NotImplementedError("PROFet response parsing not yet implemented.")


###   Synthetic energy demand model —
# fallback when the building area from Matrikkelen or the PROFet API is unavailable

SPECIFIC_PEAK_W_M2_REGULAR = {
    "House": 50.0,
    "Apartment": 40.0,
    "Office": 40.0,
    "Shop": 50.0,
    "Hotel": 55.0,
    "Kindergarten": 55.0,
    "School": 45.0,
    "University": 40.0,
    "Culture_Sport": 55.0,
    "Nursing_home": 60.0,
    "Hospital": 75.0,
    "Other": 50.0,
}

EFFICIENCY_PEAK_MULTIPLIER = {
    "Reg": 1.00,  # standard existing building
    "Eff-E": 0.75,  # efficient existing / retrofitted
    "Eff-N": 0.55,  # efficient new build
    "Vef": 0.35,  # very efficient / near-passive house
}


def make_synthetic_space_heating_timeseries(
    df_temperature: pd.DataFrame,
    usable_floor_area_m2: float,
    building_category: str,
    efficiency_key: str = "Reg",
) -> pd.DataFrame:
    """Generate a synthetic space heating demand profile for a building.

    Args:
        df_temperature: DataFrame with columns 'time' and 'temperature_C'.
        usable_floor_area_m2: Heated floor area / BRA [m²].
        building_category: PROFet building category (e.g. "House", "Office").
        efficiency_key: Energy efficiency level ("Reg", "Eff-E", "Eff-N", "Vef").

    Returns:
        DataFrame with columns 'time' and 'space_heating_kW'.
    """
    specific_peak_W_m2 = (
        SPECIFIC_PEAK_W_M2_REGULAR[building_category] * EFFICIENCY_PEAK_MULTIPLIER[efficiency_key]
    )

    df = df_temperature.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    Q_peak_space_kW = usable_floor_area_m2 * specific_peak_W_m2 / 1000.0
    heating_fraction = ((15 - df["temperature_C"]) / 35).clip(lower=0)
    df["space_heating_kW"] = Q_peak_space_kW * heating_fraction

    return df[["time", "space_heating_kW"]]


DHW_ANNUAL_KWH_DEFAULT = {
    "House": 4000.0,
    "Apartment": 40000.0,
    "Office": 10000.0,
    "Shop": 8000.0,
    "Hotel": 150000.0,
    "Kindergarten": 12000.0,
    "School": 40000.0,
    "University": 75000.0,
    "Culture_Sport": 45000.0,
    "Nursing_home": 50000.0,
    "Hospital": 750000.0,
    "Other": 10000.0,
}


def make_synthetic_dhw_timeseries(
    time: pd.Series,
    building_category: str,
) -> pd.DataFrame:
    """Generate a synthetic domestic hot water (DHW) demand profile [kW].

    Args:
        time: Series of hourly timestamps covering the period to model.
        building_category: PROFet building category (e.g. "House", "Office").

    Returns:
        DataFrame with columns 'time' and 'dhw_kW'.
    """
    dhw_annual_kWh = DHW_ANNUAL_KWH_DEFAULT[building_category]

    time = pd.to_datetime(time)
    n_hours = len(time)
    hour_of_day = time.dt.hour.to_numpy(dtype=float)

    morning_peak = np.exp(-0.5 * ((hour_of_day - 7.0) / 1.5) ** 2)
    evening_peak = 1.1 * np.exp(-0.5 * ((hour_of_day - 19.0) / 2.0) ** 2)
    shape = 0.1 + morning_peak + evening_peak
    shape = shape / shape.mean()

    dhw_average_kW = dhw_annual_kWh / n_hours

    return pd.DataFrame({"time": time, "dhw_kW": dhw_average_kW * shape})


def _make_synthetic_timeseries(
    df_temperature: pd.DataFrame,
    demand_type: DemandType,
    building_category: str,
    efficiency_key: str,
    usable_floor_area_m2: float,
) -> pd.DataFrame:
    """Synthetic hourly energy demand DataFrame; fallback when PROFet is unavailable."""
    if demand_type == "DHW":
        return make_synthetic_dhw_timeseries(
            time=df_temperature["time"],
            building_category=building_category,
        )

    if demand_type == "Space heating":
        return make_synthetic_space_heating_timeseries(
            df_temperature=df_temperature,
            usable_floor_area_m2=usable_floor_area_m2,
            building_category=building_category,
            efficiency_key=efficiency_key,
        )

    if demand_type == "Total thermal heating":
        df_dhw = make_synthetic_dhw_timeseries(
            time=df_temperature["time"],
            building_category=building_category,
        )
        df_sh = make_synthetic_space_heating_timeseries(
            df_temperature=df_temperature,
            usable_floor_area_m2=usable_floor_area_m2,
            building_category=building_category,
            efficiency_key=efficiency_key,
        )
        return pd.DataFrame(
            {
                "time": df_dhw["time"],
                "total_heating_kW": df_dhw["dhw_kW"].values + df_sh["space_heating_kW"].values,
            }
        )

    raise ValueError(f"Unknown demand_type: {demand_type!r}")
