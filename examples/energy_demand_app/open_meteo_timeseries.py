from __future__ import annotations

import pandas as pd
import requests


def fetch_temperature_open_meteo_era5_land(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m",
        "timezone": "Europe/Oslo",
        "models": "era5_land",
    }

    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()

    data = response.json()

    return pd.DataFrame(
        {
            "time": pd.to_datetime(data["hourly"]["time"]),
            "temperature_C": data["hourly"]["temperature_2m"],
        }
    )


def fetch_building_year_temperature(
    lat: float,
    lon: float,
    year: int = 2023,
) -> pd.DataFrame:
    """Fetch a full calendar year of hourly ERA5-Land 2 m temperature via Open-Meteo.

    Returns a DataFrame with columns: time, temperature_C.
    """
    return fetch_temperature_open_meteo_era5_land(
        lat=lat,
        lon=lon,
        start_date=f"{year}-01-01",
        end_date=f"{year}-12-31",
    )


if __name__ == "__main__":
    lat = 63.4305
    lon = 10.3951

    df = fetch_building_year_temperature(lat=lat, lon=lon, year=2023)

    print(df.head())
    print(df.tail())
    print(f"Hours: {len(df)}")
    print(f"Mean temperature: {df['temperature_C'].mean():.2f} °C")
    print(df["temperature_C"].tolist()[:5])
    print(df["time"].astype(str).tolist()[:5])
