from __future__ import annotations

from typing import Literal

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
) -> list[float]:
    """Extract the requested demand component from a PROFet API response.

    TODO: implement once the API response format is known.
    """
    raise NotImplementedError("PROFet response parsing not yet implemented.")


###   Making synthetic energy demand model -
# in case either the building area from Matrikkelen and/or PROFet is not accessible


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
