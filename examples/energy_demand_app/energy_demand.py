from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from PROFet_API import run_profet_for_building

DemandType = Literal["DHW", "Space heating", "Total thermal heating"]


def get_energy_demand_timeseries(
    df_temperature: pd.DataFrame,
    selected_year: int,
    building_type_code: int,
    usable_floor_area_m2: float | None,
    efficiency_key: str = "Reg",
    demand_type: DemandType = "Total thermal heating",
) -> list[float]:
    """Return an hourly energy demand time series for the given building and year.

    Tries to obtain the result from the PROFet API. Falls back to a synthetic
    time series if the API call fails (e.g. no API key configured yet).

    Args:
        df_temperature: DataFrame with columns 'time' and 'temperature_C'.
        selected_year: Calendar year to simulate.
        building_category: PROFet building category (e.g. "House", "Office").
        floor_area_m2: Heated floor area in square metres.
        efficiency_key: Energy efficiency level ("Reg", "Eff-E", "Eff-N", "Vef").
        demand_type: Which demand component to return — "DHW", "Space heating",
            or "Total thermal heating".

    Returns:
        Hourly energy demand in kWh, one value per hour (8760 or 8784 values).
    """
    try:
        result = run_profet_for_building(
            df_temperature=df_temperature,
            selected_year=selected_year,
            building_type_code=building_type_code,
            usable_floor_area_m2=usable_floor_area_m2,
            efficiency_key=efficiency_key,
        )
        return _extract_timeseries(result, demand_type)
    except Exception as exc:
        print(f"[PROFet] API call failed ({exc}), falling back to synthetic demand.")
        return _make_synthetic_timeseries(
            df_temperature=df_temperature,
            selected_year=selected_year,
            demand_type=demand_type,
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
    building_type_code: int,
    efficiency_key: str = "Reg",
) -> pd.DataFrame:
    """
    Generate a synthetic space heating demand profile for a building.

    A positive value means heating is required:
        Q_space_heating_kW > 0

    The model is:
        Q_space = Q_peak * max(0, T_balance - T_out) / (T_balance - T_low)

        with T_balance = 15 °C and T_low = -20 °C.

    Args:
        df_temperature:
            DataFrame with columns 'time' and 'temperature_C'.
        usable_floor_area_m2:
            Heated floor area / BRA [m²]. If None, a typical value is looked
            up based on building_type_code.
        building_type_code:
            Matrikkel building type code, used to determine the PROFet category
            and (if usable_floor_area_m2 is None) a typical floor area.
        efficiency_key:
            Energy efficiency level ("Reg", "Eff-E", "Eff-N", "Vef").

    Returns:
        DataFrame with columns:
            time
            space_heating_kW
    """

    from PROFet_API import usable_floor_area_m2_and_building_category

    usable_floor_area_m2, building_category = usable_floor_area_m2_and_building_category(
        usable_floor_area_m2, building_type_code
    )

    specific_peak_W_m2 = (
        SPECIFIC_PEAK_W_M2_REGULAR[building_category] * EFFICIENCY_PEAK_MULTIPLIER[efficiency_key]
    )

    df = df_temperature.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    Q_peak_space_kW = usable_floor_area_m2 * specific_peak_W_m2 / 1000.0

    heating_fraction = (15 - df["temperature_C"]) / 35

    heating_fraction = max(0, heating_fraction)

    df["space_heating_kW"] = Q_peak_space_kW * heating_fraction

    return df[
        [
            "time",
            "space_heating_kW",
        ]
    ]


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
    building_type_code: int,
) -> pd.DataFrame:
    """
    Generate a synthetic domestic hot water (DHW) demand profile [kW].

    The daily shape consists of a small base load, a morning peak around 07:00,
    and a slightly larger evening peak around 19:00. The shape is normalised so
    that the total energy over the time series equals dhw_annual_kWh, looked up
    from DHW_ANNUAL_KWH_DEFAULT based on the building type code.

    Args:
        time:
            Series of hourly timestamps covering the period to model.
        building_type_code:
            Matrikkel building type code, used to look up the PROFet category
            and the corresponding default annual DHW energy [kWh].

    Returns:
        DataFrame with columns:
            time
            dhw_kW
    """

    from PROFet_API import usable_floor_area_m2_and_building_category

    _, building_category = usable_floor_area_m2_and_building_category(None, building_type_code)

    dhw_annual_kWh = DHW_ANNUAL_KWH_DEFAULT[building_category]

    time = pd.to_datetime(time)
    n_hours = len(time)

    hour_of_day = time.dt.hour.to_numpy(dtype=float)

    morning_peak = np.exp(-0.5 * ((hour_of_day - 7.0) / 1.5) ** 2)

    evening_peak = 1.1 * np.exp(-0.5 * ((hour_of_day - 19.0) / 2.0) ** 2)

    shape = 0.1 + morning_peak + evening_peak

    # Normalise so that the mean of shape equals 1,
    # making sum(dhw_kW) equal to dhw_annual_kWh at hourly resolution.
    shape = shape / shape.mean()

    dhw_average_kW = dhw_annual_kWh / n_hours

    dhw_kW = dhw_average_kW * shape

    return pd.DataFrame(
        {
            "time": time,
            "dhw_kW": dhw_kW,
        }
    )


def _make_synthetic_timeseries(
    df_temperature: pd.DataFrame,
    selected_year: int,
    demand_type: DemandType,
) -> list[float]:
    """Generate a synthetic hourly energy demand time series.

    Used as fallback when the PROFet API is unavailable.

    TODO: implement synthetic generation based on temperature and demand_type.
    """
    raise NotImplementedError("Synthetic demand generation not yet implemented.")
