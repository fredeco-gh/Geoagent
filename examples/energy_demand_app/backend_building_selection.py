from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from fastapi import APIRouter, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pyproj import Transformer

WFS_BASE_URL = "https://wfs.geonorge.no/skwms1/wfs.matrikkelen-bygningspunkt"

# UTM zone 33 is used as the default; could be chosen dynamically based on longitude.
DEFAULT_EPSG = 25833

DEFAULT_RADIUS_M = 50.0
MAX_RADIUS_M = 200.0

_cached_typename: str | None = None
_cached_type_map: dict[str, dict] | None = None

_TYPENAME_CACHE_FILE = Path.home() / ".cache" / "jutul_agent" / "wfs_typename.txt"

_DEFAULT_TYPENAME = "app:Bygning"


def _read_typename_from_disk() -> str | None:
    try:
        if _TYPENAME_CACHE_FILE.exists():
            return _TYPENAME_CACHE_FILE.read_text().strip() or None
    except OSError:
        pass
    return None


router = APIRouter()


class BuildingInfo(BaseModel):
    bygningsnummer: str
    bygningstype: str | None = None
    bygningstype_kode: int | None = None
    bygningsstatus: str | None = None
    kommunenummer: str | None = None
    kommunenavn: str | None = None
    lat: float
    lon: float
    distance_m: float

    # Placeholders for future MatrikkelAPI extended access
    bruksareal_totalt: float | None = None
    bruksareal_til_bolig: float | None = None
    bruksareal_til_annet: float | None = None
    antall_boenheter: int | None = None


class BuildingClickResponse(BaseModel):
    hit: bool
    click_lat: float
    click_lon: float
    radius_m: float
    selected: BuildingInfo | None
    candidates: list[BuildingInfo]


app = FastAPI(title="WFS building click API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)


_wfs_caps: dict | None = None


def _get_wfs_capabilities() -> dict:
    """Call GetCapabilities once (cached per process) and return discovered WFS info."""
    global _wfs_caps
    if _wfs_caps is not None:
        return _wfs_caps

    result: dict = {"typename": None, "ns_uri": None, "output_formats": []}
    try:
        r = requests.get(
            WFS_BASE_URL,
            params={"service": "WFS", "request": "GetCapabilities"},
            timeout=30,
        )
        r.raise_for_status()
        root = ET.fromstring(r.content)

        typename: str | None = None
        for wfs_ns in ("http://www.opengis.net/wfs/2.0", "http://www.opengis.net/wfs"):
            for ft in root.findall(f".//{{{wfs_ns}}}FeatureType"):
                name_el = ft.find(f"{{{wfs_ns}}}Name")
                txt = (name_el.text or "").strip() if name_el is not None else ""
                if "bygning" in txt.lower():
                    typename = txt
            if typename:
                break

        ns_uri: str | None = None
        if typename and ":" in typename:
            prefix = typename.split(":")[0]
            m = re.search(rf'xmlns:{re.escape(prefix)}="([^"]+)"', r.text)
            if m:
                ns_uri = m.group(1)

        result = {"typename": typename, "ns_uri": ns_uri}
    except Exception as exc:
        print(f"[WFS] GetCapabilities failed: {exc}")

    _wfs_caps = result
    return _wfs_caps


def get_wfs_typename() -> str:
    """Return the WFS feature type name, discovered via GetCapabilities."""
    global _cached_typename
    if _cached_typename is not None:
        return _cached_typename

    caps = _get_wfs_capabilities()
    if caps["typename"]:
        _cached_typename = caps["typename"]
        return _cached_typename

    disk_name = _read_typename_from_disk()
    if disk_name:
        _cached_typename = disk_name
        return disk_name

    _cached_typename = _DEFAULT_TYPENAME
    return _DEFAULT_TYPENAME


def make_bbox_utm(
    lat: float,
    lon: float,
    radius_m: float,
    epsg: int = DEFAULT_EPSG,
) -> tuple[float, float, float, float]:
    """Return a square bounding box in UTM coordinates centred on (lat, lon)."""
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    x, y = transformer.transform(lon, lat)
    return x - radius_m, y - radius_m, x + radius_m, y + radius_m


def _first_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first candidate column name that exists in df, case-insensitively."""
    lower_to_actual = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        actual = lower_to_actual.get(candidate.lower())
        if actual:
            return actual
    return None


def fetch_building_points_from_wfs(
    lat: float,
    lon: float,
    radius_m: float,
    epsg: int = DEFAULT_EPSG,
) -> gpd.GeoDataFrame:
    """Query Matrikkelen Bygningspunkt WFS with a bounding box around the click point.

    Returns a GeoDataFrame of building points in the area.
    """
    caps = _get_wfs_capabilities()
    type_name = caps["typename"] or get_wfs_typename()
    minx, miny, maxx, maxy = make_bbox_utm(lat, lon, radius_m, epsg)

    params: dict[str, str] = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeNames": type_name,
        "srsName": f"EPSG:{epsg}",
        "bbox": f"{minx},{miny},{maxx},{maxy},EPSG:{epsg}",
    }
    if ":" in type_name and caps["ns_uri"]:
        prefix = type_name.split(":")[0]
        params["namespaces"] = f"xmlns({prefix},{caps['ns_uri']})"

    r = requests.get(WFS_BASE_URL, params=params, timeout=60)
    if not r.ok:
        print(f"[WFS] HTTP {r.status_code}:\n{r.text[:1000]}")
        return gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{epsg}")

    if not r.content:
        return gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{epsg}")

    # A WFS server can return HTTP 200 with an XML exception report instead of data.
    snippet = r.content[:1000].decode("utf-8", errors="replace")
    if "ExceptionReport" in snippet or "ServiceException" in snippet:
        print(f"[WFS] Server returned an exception report:\n{snippet}")
        return gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{epsg}")

    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".gml", delete=False) as tmp:
        tmp.write(r.content)
        tmp_path = Path(tmp.name)

    try:
        gdf = gpd.read_file(tmp_path)
    except (IndexError, Exception) as exc:
        print(f"[WFS] Failed to parse response ({type(exc).__name__}): {snippet[:500]}")
        return gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{epsg}")
    finally:
        tmp_path.unlink(missing_ok=True)

    if gdf.empty:
        return gdf

    return gdf.set_crs(epsg=epsg) if gdf.crs is None else gdf.to_crs(epsg=epsg)


BUILDING_TYPE_CODELIST_URL = (
    "https://register.geonorge.no/api/sosi-kodelister/kartdata/bygningstypekode.json"
)


def fetch_building_type_codelist() -> dict[str, dict]:
    """Fetch the BygningstypeKode code list from Geonorge Register.

    Returns a mapping from code → {name, description, status, id}.
    """
    response = requests.get(BUILDING_TYPE_CODELIST_URL, timeout=30)
    response.raise_for_status()

    data = response.json()
    items = data.get("containeditems", [])

    if not items:
        raise RuntimeError("No code values found in the BygningstypeKode response.")

    code_map: dict[str, dict] = {}
    for item in items:
        code = str(item.get("codevalue", "")).strip()
        if not code:
            continue
        code_map[code] = {
            "name": item.get("label"),
            "description": item.get("description"),
            "status": item.get("status"),
            "id": item.get("id"),
        }

    return code_map


def get_building_type_codelist() -> dict[str, dict]:
    """Return the cached code list, fetching from Geonorge on the first call.

    On network failure an empty dict is returned and the raw code is used as a fallback.
    """
    global _cached_type_map
    if _cached_type_map is not None:
        return _cached_type_map
    try:
        _cached_type_map = fetch_building_type_codelist()
    except Exception:
        _cached_type_map = {}
    return _cached_type_map


def get_building_type_name(code: int | str | None, code_map: dict[str, dict]) -> str | None:
    """Look up the building type name for a given code.

    Returns None if the code is missing, or the raw code string as a fallback if the
    lookup fails (e.g. the code list could not be fetched).
    """
    if code is None:
        return None
    code_str = str(code).strip()
    if not code_str:
        return None
    item = code_map.get(code_str)
    if item is None:
        # Code list unavailable or code unknown — show the raw code.
        return code_str
    return item.get("name") or code_str


def normalize_building_candidates(
    gdf: gpd.GeoDataFrame,
    click_lat: float,
    click_lon: float,
    epsg: int = DEFAULT_EPSG,
) -> list[BuildingInfo]:
    """Convert a WFS GeoDataFrame to a list of BuildingInfo, sorted by distance."""
    if gdf.empty:
        return []

    col_nr = _first_column(
        gdf,
        [
            "bygningsnummer",
            "bygningsnr",
            "bygningnummer",
            "byggnr",
            "BYGGNR",
            "bygningsNummer",
            "bygning_nummer",
        ],
    )
    if col_nr is None:
        raise RuntimeError(f"Building number column not found. WFS columns: {list(gdf.columns)}")

    col_type = _first_column(
        gdf, ["bygningstype", "bygningstypekode", "byggtype", "BYGGTYP_NBR", "bygningstypeKode"]
    )
    col_status = _first_column(
        gdf, ["bygningsstatus", "bygningstatus", "byggstatus", "BYGGSTAT", "bygningsstatusKode"]
    )
    col_kommnr = _first_column(gdf, ["kommunenummer", "kommunenr", "kommune_nr", "KOMM"])
    col_kommnm = _first_column(gdf, ["kommunenavn", "kommune", "kommune_navn"])

    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    click_x, click_y = transformer.transform(click_lon, click_lat)
    click_point = gpd.GeoSeries.from_xy([click_x], [click_y], crs=f"EPSG:{epsg}").iloc[0]

    gdf = gdf.copy()
    gdf["distance_m"] = gdf.geometry.distance(click_point)
    gdf_wgs84 = gdf.to_crs("EPSG:4326")

    type_map = get_building_type_codelist()

    def _str(row: pd.Series, col: str | None) -> str | None:
        if col is None or pd.isna(row[col]):
            return None
        return str(row[col])

    results: list[BuildingInfo] = []
    for idx, row in gdf.sort_values("distance_m").iterrows():
        point = gdf_wgs84.loc[idx].geometry.centroid
        raw_type = _str(row, col_type)
        try:
            type_kode = int(raw_type) if raw_type is not None else None
        except (ValueError, TypeError):
            type_kode = None
        results.append(
            BuildingInfo(
                bygningsnummer=str(row[col_nr]),
                bygningstype=get_building_type_name(raw_type, type_map),
                bygningstype_kode=type_kode,
                bygningsstatus=_str(row, col_status),
                kommunenummer=_str(row, col_kommnr),
                kommunenavn=_str(row, col_kommnm),
                lat=float(point.y),
                lon=float(point.x),
                distance_m=float(row["distance_m"]),
            )
        )

    return results


def get_building_info_from_wfs_click(
    lat: float,
    lon: float,
    radius_m: float = DEFAULT_RADIUS_M,
    max_candidates: int = 10,
) -> BuildingClickResponse:
    """Main entry point: click lat/lon → WFS BBOX → candidate buildings → selected building."""
    radius_m = min(float(radius_m), MAX_RADIUS_M)
    query_radius_m = max(radius_m, 15.0)

    gdf = fetch_building_points_from_wfs(lat, lon, query_radius_m)
    candidates = normalize_building_candidates(gdf, click_lat=lat, click_lon=lon)
    candidates = [c for c in candidates if c.distance_m <= radius_m][:max_candidates]
    selected = candidates[0] if candidates else None

    return BuildingClickResponse(
        hit=selected is not None,
        click_lat=lat,
        click_lon=lon,
        radius_m=radius_m,
        selected=selected,
        candidates=candidates,
    )


@router.get("/api/building-click", response_model=BuildingClickResponse)
def api_building_click(
    lat: float = Query(...),
    lon: float = Query(...),
    radius_m: float = Query(DEFAULT_RADIUS_M, ge=1, le=MAX_RADIUS_M),
    max_candidates: int = Query(10, ge=1, le=50),
) -> BuildingClickResponse:
    try:
        result = get_building_info_from_wfs_click(
            lat=lat, lon=lon, radius_m=radius_m, max_candidates=max_candidates
        )
    except requests.exceptions.RequestException as exc:
        print(f"[WFS] {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=503,
            detail={"message": f"WFS service unavailable: {exc}"},
        ) from exc
    if not result.hit:
        raise HTTPException(
            status_code=404,
            detail={
                "message": "No building point found within radius.",
                "lat": lat,
                "lon": lon,
                "radius_m": radius_m,
            },
        )
    return result


@router.get("/api/building-temperature")
def api_building_temperature(
    lat: float = Query(...),
    lon: float = Query(...),
    year: int = Query(2023, ge=1950, le=2030),
) -> dict:
    """Fetch a full year of hourly ERA5-Land 2 m temperature for a building location.

    Returns JSON with one record per hour:
      { "year": 2023, "hours": 8760, "records": [{"time_utc": ..., "temperature_C": ...}, ...] }
    """
    from open_meteo_timeseries import fetch_building_year_temperature

    df = fetch_building_year_temperature(lat=lat, lon=lon, year=year)
    records = (
        df[["time", "temperature_C"]].assign(time=df["time"].astype(str)).to_dict(orient="records")
    )
    print(f"[Open-Meteo] {len(records)} hours saved for ({lat:.5f}, {lon:.5f}), year {year}")
    return {"lat": lat, "lon": lon, "year": year, "hours": len(records), "records": records}


@router.get("/api/health")
def health() -> dict:
    return {"status": "ok", "source": "Matrikkelen Bygningspunkt WFS", "wfs": WFS_BASE_URL}


@router.get("/api/building/default-floor-area")
def api_default_floor_area(
    building_type_code: int = Query(...),
    bruksareal_totalt: float | None = Query(None),
) -> dict:
    """Return the usable floor area to use as the default for a building.

    If bruksareal_totalt is provided it is returned as-is. Otherwise the
    typical area for the building's PROFet category is returned.
    """
    from PROFet_API import usable_floor_area_m2_and_building_category

    floor_area, building_category = usable_floor_area_m2_and_building_category(
        bruksareal_totalt, building_type_code
    )
    return {"floor_area_m2": floor_area, "building_category": building_category}


def _build_temperature_report_html(bygningsnummer: str, year: int, records: list[dict]) -> str:
    import json as _json

    labels = _json.dumps([r["time"] for r in records])
    values = _json.dumps([r["temperature_C"] for r in records])
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Temperature — Building #{bygningsnummer}, {year}</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
  <style>
    body {{ font-family: sans-serif; margin: 24px; background: #f5f5f5; }}
    h1 {{ font-size: 1.1rem; color: #333; margin-bottom: 16px; }}
    .chart-wrap {{ background: #fff; border-radius: 8px; padding: 20px;
                  box-shadow: 0 1px 4px rgba(0,0,0,.1); }}
  </style>
</head>
<body>
  <h1>Temperature — Building #{bygningsnummer}, {year}</h1>
  <div class="chart-wrap"><canvas id="chart"></canvas></div>
  <script>
    new Chart(document.getElementById('chart'), {{
      type: 'line',
      data: {{
        labels: {labels},
        datasets: [{{ label: 'Temperature (°C)', data: {values},
          borderColor: '#3498db', borderWidth: 1, pointRadius: 0, tension: 0.1 }}]
      }},
      options: {{
        responsive: true,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
          x: {{ ticks: {{ maxTicksLimit: 12, maxRotation: 0 }} }},
          y: {{ title: {{ display: true, text: '°C' }} }}
        }}
      }}
    }});
  </script>
</body>
</html>"""


_temperature_report_count: dict[str, int] = {}


def make_generate_temperature_action():
    """Action handler for the energy panel's 'Generate temperature data' button."""
    import asyncio
    import uuid
    from collections.abc import Awaitable, Callable
    from typing import Any

    from jutul_agent.session import Session

    async def generate_temperature_action(
        session: Session,
        args: dict[str, Any],
        send_wire: Callable[[dict[str, Any]], Awaitable[None]],
        _queue_ui_event: Callable[[Any], None],
    ) -> None:
        lat = float(args.get("lat", 0))
        lon = float(args.get("lon", 0))
        year = int(args.get("year", 2023))
        bygningsnummer = str(args.get("bygningsnummer", "?"))
        tool_call_id = f"temp-{uuid.uuid4().hex[:8]}"

        await send_wire(
            {
                "type": "tool",
                "event": "started",
                "name": "generate_temperature",
                "label": f"Fetch temperature — building #{bygningsnummer}, {year}",
                "tool_call_id": tool_call_id,
                "args": args,
                "content": None,
            }
        )

        try:
            from open_meteo_timeseries import fetch_building_year_temperature

            df = await asyncio.get_event_loop().run_in_executor(
                None, lambda: fetch_building_year_temperature(lat=lat, lon=lon, year=year)
            )
            records = (
                df[["time", "temperature_C"]]
                .assign(time=df["time"].astype(str))
                .to_dict(orient="records")
            )

            # Energy demand — no-op until _extract_timeseries and
            # _make_synthetic_timeseries are implemented.
            bygningstype_kode = args.get("bygningstype_kode")
            bruksareal_raw = args.get("bruksareal_totalt")
            bruksareal = float(bruksareal_raw) if bruksareal_raw is not None else None
            if bygningstype_kode is not None:
                try:
                    from energy_demand import get_energy_demand_timeseries

                    _demand = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: get_energy_demand_timeseries(
                            df_temperature=df,
                            selected_year=year,
                            building_type_code=int(bygningstype_kode),
                            usable_floor_area_m2=bruksareal,
                        ),
                    )
                    # TODO: use _demand (display in UI, add to report, etc.)
                except NotImplementedError:
                    pass
                except Exception as exc:
                    print(f"[Energy demand] {exc}")

            html = _build_temperature_report_html(bygningsnummer, year, records)

            run_no = _temperature_report_count.get(session.session_id, 0) + 1
            _temperature_report_count[session.session_id] = run_no
            safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in bygningsnummer)
            rel = f"artifacts/temperature-{safe_id}-{year}-{run_no}.html"
            out = session.output_dir / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(html, encoding="utf-8")
            artifact_payload = {
                "path": rel,
                "mime": "text/html",
                "format": "html",
                "caption": f"Temperature — building #{bygningsnummer}, {year}",
                "kind": "report",
                "slot": f"temperature-{safe_id}-{year}-{run_no}",
            }
            session.trace.append("artifact", artifact_payload)
            # _run_action's send_wire does not call _flush_side_outputs after
            # "finished" events (unlike _on_message for LLM turns), so the
            # artifact would never reach the client. Forward the viz wire directly.
            from jutul_agent.interfaces.server.app import artifact_wire_events

            for wire in artifact_wire_events([artifact_payload], session.session_id):
                await send_wire(wire)

            await send_wire(
                {
                    "type": "tool",
                    "event": "finished",
                    "name": "generate_temperature",
                    "tool_call_id": tool_call_id,
                    "content": f"{len(records)} hours of temperature data ready.",
                }
            )
        except Exception as exc:
            await send_wire(
                {
                    "type": "tool",
                    "event": "finished",
                    "name": "generate_temperature",
                    "tool_call_id": tool_call_id,
                    "content": f"ERROR: {exc}",
                }
            )

    return generate_temperature_action
