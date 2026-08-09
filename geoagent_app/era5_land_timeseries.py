# DEAD CODE — not imported anywhere in the app.
# Temperature data from ERA5-Land is fetched via Open-Meteo (open_meteo_timeseries.py) instead,
# which requires no API key or local CDS client setup.

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import cdsapi
import pandas as pd

CDS_API_URL = "https://cds.climate.copernicus.eu/api"
ERA5_LAND_TIMESERIES_DATASET = "reanalysis-era5-land-timeseries"


def create_cds_client(
    api_key: str | None = None,
    api_url: str = CDS_API_URL,
) -> cdsapi.Client:
    """Create a CDS API client.

    Priority:
      1. api_key passed directly
      2. CDSAPI_KEY environment variable
      3. ~/.cdsapirc config file
    """
    if api_key is None:
        api_key = os.getenv("CDSAPI_KEY")

    if api_key:
        return cdsapi.Client(url=api_url, key=api_key)

    return cdsapi.Client()


def make_era5_land_timeseries_request(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    """Build a CDS request for ERA5-Land hourly time series at a single point.

    start_date / end_date: "YYYY-MM-DD"
    """
    return {
        "variable": ["2m_temperature"],
        "location": {
            "latitude": lat,
            "longitude": lon,
        },
        "date": [f"{start_date}/{end_date}"],
        "data_format": "netcdf",
    }


def find_time_column(df: pd.DataFrame) -> str:
    """Return the name of the time/datetime column in df."""
    candidates = ["time", "valid_time", "datetime", "date"]
    for col in candidates:
        if col in df.columns:
            return col
    for col in df.columns:
        col_norm = col.lower().strip()
        if "time" in col_norm or "date" in col_norm:
            return col
    raise ValueError(f"Time column not found. Columns: {list(df.columns)}")


def find_temperature_column(df: pd.DataFrame) -> str:
    """Return the name of the 2 m temperature column in df."""
    candidates = ["2m_temperature", "t2m", "temperature_2m"]
    for col in candidates:
        if col in df.columns:
            return col
    for col in df.columns:
        col_norm = col.lower().strip().replace(" ", "_")
        if "temperature" in col_norm or col_norm == "t2m":
            return col
    raise ValueError(f"Temperature column not found. Columns: {list(df.columns)}")


def clean_temperature_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Normalise a raw CDS response DataFrame to a clean three-column DataFrame.

    Output columns:
      time_utc        — timezone-aware UTC datetime
      temperature_K   — temperature in Kelvin
      temperature_C   — temperature in Celsius
    """
    time_col = find_time_column(df_raw)
    temp_col = find_temperature_column(df_raw)

    df = df_raw.rename(
        columns={
            time_col: "time_utc",
            temp_col: "temperature_K",
        }
    ).copy()

    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    df["temperature_K"] = pd.to_numeric(df["temperature_K"], errors="raise")
    df["temperature_C"] = df["temperature_K"] - 273.15

    df = df.sort_values("time_utc").reset_index(drop=True)
    return df[["time_utc", "temperature_K", "temperature_C"]]


def fetch_era5_land_point_temperature_timeseries(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Fetch hourly ERA5-Land 2 m temperature for a single point.

    Downloads as NetCDF, opens with xarray, and returns a tidy DataFrame.

    Returns a DataFrame with columns: time_utc, temperature_K, temperature_C.
    """
    import zipfile

    import xarray as xr

    client = create_cds_client(api_key=api_key)

    request = make_era5_land_timeseries_request(
        lat=lat,
        lon=lon,
        start_date=start_date,
        end_date=end_date,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "era5_land_temperature.nc"
        client.retrieve(ERA5_LAND_TIMESERIES_DATASET, request, str(tmp_path))

        # CDS sometimes wraps the NetCDF in a ZIP archive.
        if zipfile.is_zipfile(tmp_path):
            with zipfile.ZipFile(tmp_path) as zf:
                nc_names = [n for n in zf.namelist() if n.endswith(".nc")]
                if not nc_names:
                    raise ValueError(f"ZIP from CDS contains no .nc files: {zf.namelist()}")
                zf.extract(nc_names[0], tmpdir)
            tmp_path = Path(tmpdir) / nc_names[0]

        with xr.open_dataset(tmp_path, engine="netcdf4") as ds:
            df_raw = ds.to_dataframe().reset_index()
        df = clean_temperature_dataframe(df_raw)

    return df


def fetch_building_year_temperature(
    lat: float,
    lon: float,
    year: int = 2023,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Fetch a full calendar year of hourly ERA5-Land 2 m temperature for a building.

    lat / lon  — the building's registered coordinates (from Matrikkelen).
    year       — the calendar year to retrieve (default: 2023).

    Returns a DataFrame with columns: time_utc, temperature_K, temperature_C,
    with one row per hour (8 760 rows for a normal year, 8 784 for a leap year).
    """
    return fetch_era5_land_point_temperature_timeseries(
        lat=lat,
        lon=lon,
        start_date=f"{year}-01-01",
        end_date=f"{year}-12-31",
        api_key=api_key,
    )


def add_norwegian_local_time(df: pd.DataFrame) -> pd.DataFrame:
    """Add a Europe/Oslo local-time column to a UTC-indexed temperature DataFrame."""
    df = df.copy()
    df["time"] = df["time_utc"].dt.tz_convert("Europe/Oslo")
    return df


if __name__ == "__main__":
    # Do not commit your API key to Git.
    # Either paste it here for quick tests, or set the CDSAPI_KEY env variable.
    CDS_API_KEY = None  # or set CDSAPI_KEY env variable

    lat = 63.4305
    lon = 10.3951

    df = fetch_building_year_temperature(lat=lat, lon=lon, year=2023, api_key=CDS_API_KEY)
    df = add_norwegian_local_time(df)

    print(df.head())
    print(df.tail())
    print(f"Hours: {len(df)}")
    print(f"Mean temperature: {df['temperature_C'].mean():.2f} °C")

    temperature_series = df["temperature_C"].to_list()
    timestamps_series = df["time"].astype(str).to_list()
    print(temperature_series[:5])
    print(timestamps_series[:5])
