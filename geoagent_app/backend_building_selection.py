from __future__ import annotations

import asyncio
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
_SBHUB_BASE_URL = "https://sbhub-api.sbhub.no"


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

_MAP_TARGET = "slot:geothermal-map"


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
    bruksareal_totalt_estimated: bool | None = None  # True if any unit had zero area
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


def _estimate_total_bruksareal(areas: list[float], total_units: int) -> tuple[float | None, bool]:
    """Return (estimated_total, is_estimated).

    Fills in the average of non-zero units for any unit reporting zero area.
    is_estimated is True whenever at least one unit had zero area.
    Returns (None, False) when all areas are zero (no data at all).
    """
    nonzero = [a for a in areas if a > 0]
    if not nonzero:
        return None, False
    avg = sum(nonzero) / len(nonzero)
    return avg * total_units, len(nonzero) < total_units


def _sbhub_direct(building: BuildingInfo, headers: dict) -> BuildingInfo | None:
    """SBHub path 1: bygningsnummer → bygning_id → bruksenhet list (4 municipalities)."""
    try:
        r = requests.get(
            f"{_SBHUB_BASE_URL}/api/v1/matrikkel/bygning",
            params={"bygningsnummer": int(building.bygningsnummer), "limit": 1},
            headers=headers,
            timeout=10,
        )
    except Exception:
        return None
    if not r.ok:
        return None
    items = r.json().get("items") or []
    if not items:
        return None
    bygning_id = items[0].get("bygning_id")
    if not bygning_id:
        return None
    try:
        r2 = requests.get(
            f"{_SBHUB_BASE_URL}/api/v1/matrikkel/bruksenhet",
            params={"bygg_id": bygning_id, "limit": 1000},
            headers=headers,
            timeout=10,
        )
    except Exception:
        return None
    if not r2.ok:
        return None
    bruksenheter = r2.json().get("items") or []
    if not bruksenheter:
        return None
    areas = [float(be.get("bruksareal") or 0) for be in bruksenheter]
    totalt, estimated = _estimate_total_bruksareal(areas, len(bruksenheter))
    print(f"[SBHub] direct: {len(bruksenheter)} units, totalt={totalt}, estimated={estimated}")
    return building.model_copy(
        update={
            "bruksareal_totalt": totalt,
            "bruksareal_totalt_estimated": estimated if totalt is not None else None,
            "antall_boenheter": len(bruksenheter),
        }
    )


def _sbhub_via_address(building: BuildingInfo, headers: dict) -> BuildingInfo | None:
    """SBHub path 2: obtain street address from building location → address search → bruksenheter

    Uses the house_number and house_letter from the search response to isolate units
    belonging to the specific clicked building, not other buildings on the same parcel.
    """
    import concurrent.futures

    # Step 1: obtain street address from building location via Geonorge punktsok
    try:
        rg = requests.get(
            "https://ws.geonorge.no/adresser/v1/punktsok",
            params={"lat": building.lat, "lon": building.lon, "radius": 50, "treffPerSide": 1},
            timeout=10,
        )
    except Exception as exc:
        print(f"[SBHub] punktsok failed: {exc}")
        return None
    if not rg.ok:
        print(f"[SBHub] punktsok -> HTTP {rg.status_code}")
        return None
    adresser = rg.json().get("adresser") or []
    if not adresser:
        print("[SBHub] punktsok: no address found near building")
        return None
    a = adresser[0]
    nummer = str(a.get("nummer") or "")
    bokstav = (a.get("bokstav") or "").upper()
    addr = (
        f"{a.get('adressenavn', '')} {nummer}"
        f"{bokstav}, {a.get('postnummer', '')} {a.get('poststed', '')}"
    )
    print(f"[SBHub] address path: {addr}")

    # Step 2: address → matrikkelenhet_id
    try:
        ra = requests.get(
            f"{_SBHUB_BASE_URL}/api/v1/elhub-consent/matrikkelunit/address",
            params={"address": addr, "limit": 1},
            headers=headers,
            timeout=10,
        )
    except Exception as exc:
        print(f"[SBHub] address search failed: {exc}")
        return None
    if not ra.ok:
        print(f"[SBHub] address search -> HTTP {ra.status_code}")
        return None
    ra_results = ra.json().get("results") or []
    if not ra_results:
        print("[SBHub] address search: no matrikkelenhet found")
        return None
    matrikkelenhet_id = ra_results[0].get("matrikkelenhet_id")
    if not matrikkelenhet_id:
        return None
    print(f"[SBHub] matrikkelenhet_id={matrikkelenhet_id}")

    # Step 3 (fast path): matrikkel/bruksenhet?matrikkelenhet_id=X — one request,
    # bruksareal included. Tried for all municipalities; fails fast with a short
    # timeout for those where the endpoint has no data.
    try:
        rm = requests.get(
            f"{_SBHUB_BASE_URL}/api/v1/matrikkel/bruksenhet",
            params={"matrikkelenhet_id": matrikkelenhet_id, "limit": 1000},
            headers=headers,
            timeout=5,
        )
        if rm.ok:
            matrikkel_items = rm.json().get("items") or []
            if matrikkel_items:
                bygg_ids = {it.get("bygg_id") for it in matrikkel_items if it.get("bygg_id")}
                if len(bygg_ids) == 1:
                    areas = [float(it.get("bruksareal") or 0) for it in matrikkel_items]
                    totalt, estimated = _estimate_total_bruksareal(areas, len(matrikkel_items))
                    print(
                        f"[SBHub] matrikkel fast path: {len(matrikkel_items)} units,"
                        f" totalt={totalt}, estimated={estimated}"
                    )
                    return building.model_copy(
                        update={
                            "bruksareal_totalt": totalt,
                            "bruksareal_totalt_estimated": estimated
                            if totalt is not None
                            else None,
                            "antall_boenheter": len(matrikkel_items),
                        }
                    )
                print(
                    f"[SBHub] matrikkel fast path: {len(bygg_ids)} bygg_ids on parcel"
                    f" — falling through to bruksenhet-search"
                )
    except Exception as exc:
        print(f"[SBHub] matrikkel fast path failed: {exc}")

    # Step 4: bruksenhet-search filtered by house_number + house_letter to isolate
    # units belonging to the clicked building (not neighbours on the same parcel).
    try:
        rb = requests.get(
            f"{_SBHUB_BASE_URL}/api/v1/elhub-consent/matrikkelunit/bruksenhet-search",
            params={"matrikkelenhet_id": matrikkelenhet_id},
            headers=headers,
            timeout=10,
        )
    except Exception as exc:
        print(f"[SBHub] bruksenhet-search failed: {exc}")
        return None
    if not rb.ok:
        print(f"[SBHub] bruksenhet-search -> HTTP {rb.status_code}")
        return None
    all_items = rb.json().get("results") or []
    unique_numbers = sorted(
        {
            (str(it.get("house_number") or ""), (it.get("house_letter") or "").upper())
            for it in all_items
        }
    )
    print(
        f"[SBHub] bruksenhet-search: {len(all_items)} units on parcel,"
        f" looking for ({nummer!r}, {bokstav!r}), found: {unique_numbers}"
    )
    be_ids = list(
        dict.fromkeys(
            item["bruksenhet_id"]
            for item in all_items
            if item.get("bruksenhet_id")
            and str(item.get("house_number") or "") == nummer
            and (item.get("house_letter") or "").upper() == bokstav
        )
    )
    if not be_ids:
        print("[SBHub] no bruksenheter matching this building's address")
        return None
    raw_count = sum(
        1
        for item in all_items
        if item.get("bruksenhet_id")
        and str(item.get("house_number") or "") == nummer
        and (item.get("house_letter") or "").upper() == bokstav
    )
    print(
        f"[SBHub] {len(be_ids)} unique bruksenhet(er) for this building"
        f" ({raw_count} raw, {len(all_items)} total on parcel)"
    )

    # Fetch bruksareal for each unit in parallel via the elhub-consent endpoint.
    def _fetch(beid: int) -> dict | None:
        try:
            rf = requests.get(
                f"{_SBHUB_BASE_URL}/api/v1/elhub-consent/matrikkelunit/bruksenhet/{beid}",
                headers=headers,
                timeout=50,
            )
            if rf.ok:
                return rf.json()
            print(f"[SBHub] elhub bruksenhet/{beid} -> HTTP {rf.status_code}")
            return None
        except Exception as exc:
            print(f"[SBHub] elhub bruksenhet/{beid} failed: {exc}")
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        fetched = [r for r in pool.map(_fetch, be_ids) if r is not None]

    if not fetched:
        print(f"[SBHub] all {len(be_ids)} individual bruksenhet fetches failed")
        return None
    areas = [float(be.get("bruksareal") or 0) for be in fetched]
    totalt, estimated = _estimate_total_bruksareal(areas, len(be_ids))
    print(
        f"[SBHub] {len(fetched)}/{len(be_ids)} units fetched,"
        f" totalt={totalt}, estimated={estimated}"
    )
    return building.model_copy(
        update={
            "bruksareal_totalt": totalt,
            "bruksareal_totalt_estimated": estimated if totalt is not None else None,
            "antall_boenheter": len(be_ids),
        }
    )


def _enrich_with_sbhub(building: BuildingInfo) -> BuildingInfo:
    """Fetch bruksareal from the SBHub Matrikkel API and populate the building's area fields.

    Reads the API token from the SBHUB_API_TOKEN environment variable.
    Returns the original building unchanged if the token is missing or the API call fails.
    """
    import os

    token = os.environ.get("SBHUB_API_TOKEN")
    if not token:
        print("[SBHub] SBHUB_API_TOKEN not set — skipping bruksareal lookup")
        return building

    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    print(f"[SBHub] enriching bygningsnummer={building.bygningsnummer}")

    try:
        result = _sbhub_direct(building, headers) or _sbhub_via_address(building, headers)
        return result if result is not None else building
    except Exception as exc:
        print(f"[SBHub] failed to enrich building {building.bygningsnummer}: {exc}")
        return building


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


def _parse_building_gml(content: bytes, bygningsnummer: str, epsg: int) -> BuildingInfo | None:
    """Parse a WFS GML response and return the BuildingInfo for the requested number.

    Returns None if the response is empty, unparseable, or contains no feature
    whose building number column matches *bygningsnummer*.
    """
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".gml", delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        gdf = gpd.read_file(tmp_path)
    except Exception as exc:
        print(f"[WFS] failed to parse building lookup response: {exc}")
        return None
    finally:
        tmp_path.unlink(missing_ok=True)

    if gdf.empty:
        return None

    gdf = gdf.set_crs(epsg=epsg) if gdf.crs is None else gdf.to_crs(epsg=epsg)

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
    # If server-side filtering worked we have exactly one row; otherwise filter
    # client-side to find the row whose building number matches.
    if col_nr is not None:
        mask = gdf[col_nr].astype(str) == str(bygningsnummer)
        gdf = gdf[mask]
        if gdf.empty:
            return None

    gdf_wgs84 = gdf.to_crs("EPSG:4326")
    row = gdf.iloc[0]
    point = gdf_wgs84.iloc[0].geometry.centroid

    col_type = _first_column(
        gdf, ["bygningstype", "bygningstypekode", "byggtype", "BYGGTYP_NBR", "bygningstypeKode"]
    )
    col_status = _first_column(
        gdf, ["bygningsstatus", "bygningstatus", "byggstatus", "BYGGSTAT", "bygningsstatusKode"]
    )
    col_kommnr = _first_column(gdf, ["kommunenummer", "kommunenr", "kommune_nr", "KOMM"])
    col_kommnm = _first_column(gdf, ["kommunenavn", "kommune", "kommune_navn"])

    def _str(col: str | None) -> str | None:
        if col is None or col not in row.index or pd.isna(row[col]):
            return None
        return str(row[col])

    raw_type = _str(col_type)
    try:
        type_kode = int(raw_type) if raw_type is not None else None
    except (ValueError, TypeError):
        type_kode = None

    type_map = get_building_type_codelist()
    return BuildingInfo(
        bygningsnummer=_str(col_nr) or bygningsnummer,
        bygningstype=get_building_type_name(raw_type, type_map),
        bygningstype_kode=type_kode,
        bygningsstatus=_str(col_status),
        kommunenummer=_str(col_kommnr),
        kommunenavn=_str(col_kommnm),
        lat=float(point.y),
        lon=float(point.x),
        distance_m=0.0,
    )


def fetch_building_by_number(
    bygningsnummer: str,
    epsg: int = DEFAULT_EPSG,
) -> BuildingInfo | None:
    """Look up a single building by its bygningsnummer in the Matrikkelen WFS.

    Tries three strategies in order:
    1. FEATUREID lookup (WFS 2.0 standard, most reliable).
    2. CQL_FILTER (fast if supported by the server).
    3. OGC XML Filter (standard fallback, no count limit so may be slow).

    Returns None if no matching building is found or the WFS is unavailable.
    """
    caps = _get_wfs_capabilities()
    type_name = caps["typename"] or get_wfs_typename()
    base_params: dict[str, str] = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "srsName": f"EPSG:{epsg}",
    }
    if ":" in type_name and caps["ns_uri"]:
        prefix = type_name.split(":")[0]
        base_params["namespaces"] = f"xmlns({prefix},{caps['ns_uri']})"

    def _get_gml(extra: dict[str, str]) -> bytes | None:
        params = {**base_params, **extra}
        try:
            r = requests.get(WFS_BASE_URL, params=params, timeout=60)
        except Exception as exc:
            print(f"[WFS] request failed: {exc}")
            return None
        if not r.ok or not r.content:
            return None
        snippet = r.content[:500].decode("utf-8", errors="replace")
        if "ExceptionReport" in snippet or "ServiceException" in snippet:
            print(f"[WFS] server exception for {extra}: {snippet[:300]}")
            return None
        return r.content

    # Strategy 1: FEATUREID — WFS 2.0 fetches a feature by its GML @gml:id.
    # Matrikkelen uses the pattern "<prefix>:Bygning.<bygningsnummer>".
    type_local = type_name.split(":")[-1]  # e.g. "Bygning" from "app:Bygning"
    prefix = type_name.split(":")[0] if ":" in type_name else ""
    for fid in (
        f"{prefix}:{type_local}.{bygningsnummer}" if prefix else None,
        f"{type_local}.{bygningsnummer}",
    ):
        if fid is None:
            continue
        content = _get_gml({"FEATUREID": fid})
        if content:
            result = _parse_building_gml(content, bygningsnummer, epsg)
            if result is not None:
                return result

    # Strategy 2: CQL_FILTER — fast attribute filter (not all servers support it).
    content = _get_gml(
        {
            "typeNames": type_name,
            "CQL_FILTER": f"bygningsnummer='{bygningsnummer}'",
            "count": "100",
        }
    )
    if content:
        result = _parse_building_gml(content, bygningsnummer, epsg)
        if result is not None:
            return result

    # Strategy 3: OGC XML Filter — standard WFS 2.0 attribute query.
    filter_xml = (
        "<Filter>"
        "<PropertyIsEqualTo>"
        "<ValueReference>bygningsnummer</ValueReference>"
        f"<Literal>{bygningsnummer}</Literal>"
        "</PropertyIsEqualTo>"
        "</Filter>"
    )
    content = _get_gml(
        {
            "typeNames": type_name,
            "FILTER": filter_xml,
            "count": "100",
        }
    )
    if content:
        result = _parse_building_gml(content, bygningsnummer, epsg)
        if result is not None:
            return result

    return None


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


@router.get("/api/building/floor-area")
def api_building_floor_area(
    bygningsnummer: str = Query(...),
    lat: float = Query(...),
    lon: float = Query(...),
    kommunenummer: str | None = Query(None),
) -> dict:
    """Look up bruksareal for a building via SBHub. Called by the 'Look up floor area' button."""
    building = BuildingInfo(
        bygningsnummer=bygningsnummer,
        lat=lat,
        lon=lon,
        distance_m=0.0,
        kommunenummer=kommunenummer,
    )
    enriched = _enrich_with_sbhub(building)
    if enriched.bruksareal_totalt is None:
        raise HTTPException(status_code=404, detail="Could not retrieve floor area from SBHub.")
    return {
        "bruksareal_totalt": enriched.bruksareal_totalt,
        "bruksareal_totalt_estimated": enriched.bruksareal_totalt_estimated,
        "antall_boenheter": enriched.antall_boenheter,
    }


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


def _build_energy_demand_report_html(
    bygningsnummer: str,
    year: int,
    demand_type: str,
    records: list[dict],
    borehole_records: list[dict] | None = None,
) -> str:
    import json as _json

    labels = _json.dumps([r["time"] for r in records])
    demand_values = _json.dumps([round(r["kW"], 2) for r in records])

    extra_canvases = ""
    extra_chart_calls = ""
    if borehole_records is not None:
        t_in_vals = _json.dumps([round(r["T_in"], 2) for r in borehole_records])
        t_out_vals = _json.dumps([round(r["T_out"], 2) for r in borehole_records])
        dt_vals = _json.dumps([round(r["T_out"] - r["T_in"], 2) for r in borehole_records])
        extra_canvases = (
            '  <div class="chart-wrap"><canvas id="chart-tin"></canvas></div>\n'
            '  <div class="chart-wrap"><canvas id="chart-tout"></canvas></div>\n'
            '  <div class="chart-wrap"><canvas id="chart-dt"></canvas></div>'
        )
        extra_chart_calls = (
            f"    makeChart('chart-tin', {t_in_vals}, '#2980b9', 'T_in [°C]');\n"
            f"    makeChart('chart-tout', {t_out_vals}, '#27ae60', 'T_out [°C]');\n"
            f"    makeChart('chart-dt', {dt_vals}, '#8e44ad', 'T_out - T_in [°C]');"
        )

    title = f"Building #{bygningsnummer} — {year}"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{title}</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
  <style>
    body {{ font-family: sans-serif; margin: 24px; background: #f5f5f5; }}
    h1 {{ font-size: 1.1rem; color: #333; margin-bottom: 16px; }}
    .chart-wrap {{ background: #fff; border-radius: 8px; padding: 20px;
                  box-shadow: 0 1px 4px rgba(0,0,0,.1); margin-bottom: 24px; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div class="chart-wrap"><canvas id="chart-demand"></canvas></div>
{extra_canvases}
  <script>
    const LABELS = {labels};
    const MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
    function fmtDate(s) {{
      const d = new Date(s);
      return isNaN(d) ? s : String(d.getDate()).padStart(2,'0') + ' ' + MONTHS[d.getMonth()];
    }}
    function makeChart(id, data, color, yTitle) {{
      new Chart(document.getElementById(id), {{
        type: 'line',
        data: {{
          labels: LABELS,
          datasets: [{{ data, borderColor: color, borderWidth: 1.5,
            pointRadius: 0, tension: 0.1, fill: false }}]
        }},
        options: {{
          responsive: true,
          animation: false,
          plugins: {{ legend: {{ display: false }} }},
          scales: {{
            x: {{
              title: {{ display: true, text: 'Date' }},
              ticks: {{
                maxTicksLimit: 12, maxRotation: 0,
                callback(v) {{ return fmtDate(this.getLabelForValue(v)); }}
              }}
            }},
            y: {{ title: {{ display: true, text: yTitle }} }}
          }}
        }}
      }});
    }}
    makeChart('chart-demand', {demand_values}, '#e67e22', '{demand_type} [kW]');
{extra_chart_calls}
  </script>
</body>
</html>"""


_energy_demand_run_count: dict[str, int] = {}

# Most-recent energy demand data per session, written by generate_energy_demands
# so the agent's run_borehole_simulation tool can pick it up without the user
# having to re-enter the same demand series.
_session_demand_data: dict[str, dict] = {}


def get_session_demand_data(session_id: str) -> dict | None:
    """Return the most-recent energy demand data for a session, or None."""
    return _session_demand_data.get(session_id)


# Warmup template: pre-compiles PythonCall and loads pygfunction_sim.jl into
# the Julia kernel during session startup so the first agent call is fast.
_PYGSIM_WARMUP_TEMPLATE = """
try
    ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
    ENV["JULIA_PYTHONCALL_EXE"] = raw"__PYTHON_EXE__"
    if !isdefined(Main, :run_borehole_simulation)
        include(raw"__PYGSIM_JL_PATH__")
    end
catch
end
"""

# Holds references to warmup tasks so asyncio doesn't GC them mid-flight.
_pygsim_warmup_tasks: set[asyncio.Task[None]] = set()


def start_pygsim_warmup(session: object, pygsim_jl_path: str) -> None:
    """Fire-and-forget: pre-compile PythonCall and load pygfunction_sim.jl.

    Shifts the JIT cost from the first agent call to session startup, where
    it runs alongside the Fimbul warmup and the user already expects a delay.
    Retried for up to 30 s in case the kernel is still busy with the Fimbul
    warmup when this fires.
    """
    import sys

    python_exe = sys.executable.replace("\\", "/")
    jl_path = Path(pygsim_jl_path).resolve().as_posix()
    code = _PYGSIM_WARMUP_TEMPLATE.replace("__PYTHON_EXE__", python_exe).replace(
        "__PYGSIM_JL_PATH__", jl_path
    )

    async def _run() -> None:
        for attempt in range(30):
            try:
                await session.julia.eval(code)  # type: ignore[attr-defined]
                return
            except Exception:
                if attempt == 29:
                    return  # best-effort
                await asyncio.sleep(1.0)

    task = asyncio.create_task(_run())
    _pygsim_warmup_tasks.add(task)
    task.add_done_callback(_pygsim_warmup_tasks.discard)


# Julia template that runs the pygfunction borehole simulation via the
# PythonCall overlay (examples/geoagent_app/julia/pygfunction_sim.jl).
# Paths are injected by make_generate_energy_demands_action at call time.
_PYGSIM_TEMPLATE = """
begin
    ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
    ENV["JULIA_PYTHONCALL_EXE"] = raw"__PYTHON_EXE__"
    if !isdefined(Main, :run_borehole_simulation)
        include(raw"__PYGSIM_JL_PATH__")
    end
    import JSON3
    local _inp    = JSON3.read(read(raw"__INPUT_PATH__", String))
    local _ts     = [String(v)  for v in _inp["timestamps"]]
    local _kw     = [Float64(v) for v in _inp["demand_kw"]]
    local _result = run_borehole_simulation(_ts, _kw;
        lat = Float64(_inp["lat"]),
        lon = Float64(_inp["lon"]),
    )
    open(raw"__OUTPUT_PATH__", "w") do io
        JSON3.write(io, _result)
    end
    "ok"
end
"""


def make_generate_energy_demands_action():
    """Action handler for the energy panel's 'Generate heating needs' button."""
    import uuid
    from collections.abc import Awaitable, Callable
    from typing import Any

    from jutul_agent.session import Session

    async def generate_energy_demands_action(
        session: Session,
        args: dict[str, Any],
        send_wire: Callable[[dict[str, Any]], Awaitable[None]],
        _queue_ui_event: Callable[[Any], None],
    ) -> None:
        lat = float(args.get("lat", 0))
        lon = float(args.get("lon", 0))
        year = int(args.get("year", 2023))
        bygningsnummer = str(args.get("bygningsnummer", "?"))
        bygningstype_kode = args.get("bygningstype_kode")
        building_category = str(args.get("building_category") or "") or None
        raw_area = args.get("usable_floor_area_m2")
        usable_floor_area_m2 = float(raw_area) if raw_area is not None else None
        efficiency_key = str(args.get("efficiency_key") or "Reg")
        demand_type = str(args.get("demand_type") or "Total thermal heating")
        tool_call_id = f"energy-{uuid.uuid4().hex[:8]}"

        await send_wire(
            {
                "type": "tool",
                "event": "started",
                "name": "generate_energy_demands",
                "label": (
                    f"Compute {demand_type.lower()} + borehole temperatures"
                    f" — building #{bygningsnummer}, {year}"
                ),
                "tool_call_id": tool_call_id,
                "args": args,
                "content": None,
            }
        )

        try:
            from energy_demand import get_energy_demand_timeseries
            from open_meteo_timeseries import fetch_building_year_temperature

            df_temp = await asyncio.get_event_loop().run_in_executor(
                None, lambda: fetch_building_year_temperature(lat=lat, lon=lon, year=year)
            )

            if bygningstype_kode is None and not building_category:
                raise ValueError("Building type code missing — click a building on the map first.")

            df_demand = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: get_energy_demand_timeseries(
                    df_temperature=df_temp,
                    selected_year=year,
                    building_type_code=int(bygningstype_kode)
                    if bygningstype_kode is not None
                    else None,
                    usable_floor_area_m2=usable_floor_area_m2,
                    efficiency_key=efficiency_key,
                    demand_type=demand_type,
                    building_category=building_category,
                ),
            )

            # Normalize to a "kW" column regardless of which demand type was used
            demand_col = next(c for c in df_demand.columns if c != "time")
            records = (
                df_demand[["time", demand_col]]
                .rename(columns={demand_col: "kW"})
                .assign(time=df_demand["time"].astype(str))
                .to_dict(orient="records")
            )

            # Persist demand data for the agent's run_borehole_simulation tool.
            _session_demand_data[session.session_id] = {
                "timestamps": [str(ts) for ts in df_demand["time"]],
                "demand_kw": df_demand[demand_col].tolist(),
                "lat": lat,
                "lon": lon,
                "bygningsnummer": bygningsnummer,
                "year": year,
                "demand_type": demand_type,
            }
            # The model wasn't involved in this action, so fold a note into
            # whatever the user sends next so the agent knows which building's
            # demand data is now loaded (same mechanism as well clicks and
            # simulationCompleted in make_run_simulation_action).
            _queue_ui_event(
                {
                    "event": "HeatingNeedsGenerated",
                    "bygningsnummer": bygningsnummer,
                    "year": year,
                    "demand_type": demand_type,
                    "n_hours": len(df_demand),
                }
            )

            html = _build_energy_demand_report_html(bygningsnummer, year, demand_type, records)

            safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in bygningsnummer)
            run_no = _energy_demand_run_count.get(session.session_id + safe_id, 0) + 1
            _energy_demand_run_count[session.session_id + safe_id] = run_no
            rel = f"artifacts/energy-demand-{safe_id}-{run_no}.html"
            out = session.output_dir / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(html, encoding="utf-8")
            artifact_payload = {
                "path": rel,
                "mime": "text/html",
                "format": "html",
                "caption": f"{demand_type} — building #{bygningsnummer}, {year}",
                "kind": "report",
                "slot": f"energy-demand-{safe_id}",
            }
            session.trace.append("artifact", artifact_payload)
            # _run_action's send_wire does not flush side outputs after "finished"
            # events, so forward the viz wire directly.
            from jutul_agent.interfaces.server.app import artifact_wire_events

            for wire in artifact_wire_events([artifact_payload], session.session_id):
                await send_wire(wire)

            await send_wire(
                {
                    "type": "tool",
                    "event": "finished",
                    "name": "generate_energy_demands",
                    "tool_call_id": tool_call_id,
                    "content": (
                        f"{len(records)} hours of {demand_type.lower()} data"
                        " + borehole temperatures ready."
                    ),
                }
            )
            await send_wire(
                {
                    "type": "ui",
                    "action": "energy_demand_ready",
                    "payload": {"status": "done"},
                    "target": _MAP_TARGET,
                }
            )
        except Exception as exc:
            await send_wire(
                {
                    "type": "tool",
                    "event": "finished",
                    "name": "generate_energy_demands",
                    "tool_call_id": tool_call_id,
                    "content": f"ERROR: {exc}",
                }
            )
            await send_wire(
                {
                    "type": "ui",
                    "action": "energy_demand_ready",
                    "payload": {"status": "error", "message": str(exc)},
                    "target": _MAP_TARGET,
                }
            )

    return generate_energy_demands_action
