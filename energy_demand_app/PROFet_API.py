from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests

PROFET_BASE_URL = "https://flexibilitysuite.byggforsk.no/index.html"
PROFET_VALIDATE_URL = f"{PROFET_BASE_URL}/api/Profet/Validate"
PROFET_RUN_URL = f"{PROFET_BASE_URL}/api/Profet"

token = os.getenv("PROFET_API_TOKEN")

PROFET_BUILDING_CATEGORIES = {
    "House",
    "Apartment",
    "Office",
    "Shop",
    "Hotel",
    "Kindergarten",
    "School",
    "University",
    "Culture_Sport",
    "Nursing_home",
    "Hospital",
    "Other",
}

TYPICAL_AREAS_M2 = {
    "House": 160.0,
    "Apartment": 1800.0,
    "Office": 2000.0,
    "Shop": 1000.0,
    "Hotel": 5000.0,
    "Kindergarten": 700.0,
    "School": 5000.0,
    "University": 15000.0,
    "Culture_Sport": 3000.0,
    "Nursing_home": 5000.0,
    "Hospital": 30000.0,
    "Other": 1000.0,
}


def map_matrikkel_building_type_to_profet_category(
    building_type_code: int | str,
) -> str:
    code = int(building_type_code)

    # Småhus, enebolig, tomannsbolig, rekkehus
    if 111 <= code <= 136:
        return "House"

    # Store boligbygg / blokker
    if 141 <= code <= 159 or code == 523:
        return "Apartment"

    # Hotell / overnatting
    if 511 <= code <= 529 and code != 523:
        return "Hotel"

    # Kontor
    if 311 <= code <= 319 and code != 313:
        return "Office"

    # Butikk / forretning
    if 321 <= code <= 329:
        return "Shop"

    # Skole / universitet
    if 611 <= code <= 619:
        return "School"

    if 621 <= code <= 629 and code != 623:
        return "University"

    # Barnehage
    if code == 612:
        return "Kindergarten"

    # Sykehus / institusjon
    if 719 <= code <= 739:
        return "Hospital"

    # Sykehjem / omsorg
    if 721 <= code <= 729:
        return "Nursing_home"

    if code == 611 or 641 <= code <= 669:
        return "Culture_Sport"

    return "Other"


def usable_floor_area_m2_and_building_category(
    usable_floor_area_m2: float | None, building_type_code: int
) -> tuple[float, str]:

    building_category = map_matrikkel_building_type_to_profet_category(building_type_code)

    if usable_floor_area_m2 is None:
        usable_floor_area_m2 = TYPICAL_AREAS_M2[building_category]

    return usable_floor_area_m2, building_category


PROFET_EFFICIENCY_KEYS = {
    "Reg",  # Regular
    "Eff-E",  # Efficient existing / rehabilitated
    "Eff-N",  # Efficient new
    "Vef",  # Very efficient
}


def build_area_distribution(
    usable_floor_area_m2: float,
    efficiency_key: str = "Reg",
) -> dict[str, float]:
    """
    Build an area distribution across efficiency levels.

    For a simple use case the entire floor area is placed in one level
    (typically "Reg") and the others are set to 0.
    """

    if usable_floor_area_m2 <= 0:
        raise ValueError("floor_area_m2 must be greater than 0.")

    if efficiency_key not in PROFET_EFFICIENCY_KEYS:
        raise ValueError(
            f"Invalid efficiency_key: {efficiency_key}. "
            f"Must be one of {sorted(PROFET_EFFICIENCY_KEYS)}"
        )

    area_distribution = {
        "Reg": 0.0,
        "Eff-E": 0.0,
        "Eff-N": 0.0,
        "Vef": 0.0,
    }

    area_distribution[efficiency_key] = float(usable_floor_area_m2)

    return area_distribution


def build_areas_for_one_building(
    building_category: str,
    usable_floor_area_m2: float,
    efficiency_key: str = "Reg",
) -> dict[str, dict[str, float]]:
    """
    Build the Areas section of a PROFet payload.

    Example:
      building_category = "Office"
      floor_area_m2 = 1200
      efficiency_key = "Reg"

    gives:
      {
        "Office": {
          "Reg": 1200,
          "Eff-E": 0,
          "Eff-N": 0,
          "Vef": 0
        }
      }
    """

    if building_category not in PROFET_BUILDING_CATEGORIES:
        raise ValueError(
            f"Invalid PROFet category: {building_category}. "
            f"Must be one of {sorted(PROFET_BUILDING_CATEGORIES)}"
        )

    return {
        building_category: build_area_distribution(
            usable_floor_area_m2=usable_floor_area_m2,
            efficiency_key=efficiency_key,
        )
    }


def build_tout_from_dataframe(
    df_temperature: pd.DataFrame,
    selected_year: int,
    time_col: str = "time",
    temperature_col: str = "temperature_C",
) -> tuple[str, list[float]]:
    """
    Extract a full year of hourly temperatures from a DataFrame.

    Expected columns:
      time
      temperature_C

    Returns:
      StartDate = YYYY-01-01
      Tout = hourly temperatures
    """

    df = df_temperature.copy()

    df[time_col] = pd.to_datetime(df[time_col])

    df_year = df[df[time_col].dt.year == selected_year].copy()
    df_year = df_year.sort_values(time_col).reset_index(drop=True)

    if df_year.empty:
        raise ValueError(f"No temperature data found for year {selected_year}.")

    first_time = df_year.loc[0, time_col]

    if first_time.month != 1 or first_time.day != 1 or first_time.hour != 0:
        raise ValueError(
            f"Temperature series for {selected_year} must start on 1 January at 00:00. "
            f"First timestamp is {first_time}."
        )

    temperatures_C = df_year[temperature_col].astype(float).to_list()

    start_date = f"{selected_year}-01-01"

    if not temperatures_C:
        raise ValueError("temperatures_C is empty.")

    if len(temperatures_C) not in {8760, 8784}:
        raise ValueError(
            f"Expected 8760 hours for a regular year or 8784 for a leap year. "
            f"Got {len(temperatures_C)} temperature values."
        )

    return start_date, temperatures_C


def build_profet_v2_payload(
    df_temperature: pd.DataFrame,
    selected_year: int,
    building_category: str,
    usable_floor_area_m2: float,
    efficiency_key: str = "Reg",
) -> dict[str, Any]:
    """
    Build a PROFet v2 payload based on the sample request structure:

      StartDate
      Areas
      TimeSeries.Tout
      RetInd
      Country

    Uses human-readable category names: "Office", "House", "Apartment", etc.
    """

    start_date, tout = build_tout_from_dataframe(
        df_temperature=df_temperature,
        selected_year=selected_year,
        time_col="time",
        temperature_col="temperature_C",
    )

    areas = build_areas_for_one_building(
        building_category=building_category,
        usable_floor_area_m2=usable_floor_area_m2,
        efficiency_key=efficiency_key,
    )

    payload = {
        "StartDate": start_date,
        "Areas": areas,
        "TimeSeries": {
            "Tout": tout,
        },
        "RetInd": True,
        "Country": "Norway",
    }

    return payload


def post_json(
    url: str,
    payload: dict[str, Any],
    timeout: int = 180,
    token: str | None = token,
) -> dict[str, Any]:

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
    }

    response = requests.post(
        url,
        json=payload,
        headers=headers,
        timeout=timeout,
    )

    if response.status_code == 401:
        raise RuntimeError(
            "PROFet API responded 401 Unauthorized.\n"
            "That means the Bearer-token is missing, wrong or expired, "
            "or that your account does not have access to this API.\n"
            f"Response headers: {dict(response.headers)}"
        )

    if not response.ok:
        raise RuntimeError(
            f"PROFet API error\n"
            f"URL: {url}\n"
            f"Status: {response.status_code}\n"
            f"Response:\n{response.text}"
        )

    return response.json()


def validate_profet_payload(payload: dict[str, Any], token: str | None = token) -> dict[str, Any]:
    return post_json(url=PROFET_VALIDATE_URL, payload=payload, timeout=60, token=token)


def run_profet(payload: dict[str, Any], token: str | None = token) -> dict[str, Any]:
    return post_json(url=PROFET_RUN_URL, payload=payload, timeout=180, token=token)


def run_profet_for_building(
    df_temperature: pd.DataFrame,
    selected_year: int,
    building_category: str,
    usable_floor_area_m2: float,
    efficiency_key: str = "Reg",
    token: str | None = token,
) -> dict[str, Any]:
    """
    Full PROFet flow:
      1. build payload
      2. validate payload
      3. run PROFet
      4. return raw JSON response
    """

    payload = build_profet_v2_payload(
        df_temperature=df_temperature,
        selected_year=selected_year,
        building_category=building_category,
        usable_floor_area_m2=usable_floor_area_m2,
        efficiency_key=efficiency_key,
    )

    validate_profet_payload(payload, token)
    result = run_profet(payload, token)

    return result
