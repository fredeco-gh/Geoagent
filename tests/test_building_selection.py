"""Unit tests for SBHub bruksareal enrichment in backend_building_selection.

All HTTP calls are mocked — no real network or Julia runtime is needed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "geoagent_app" / "backend_building_selection.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("backend_building_selection", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bbs = _load_module()
BuildingInfo = bbs.BuildingInfo
_sbhub_direct = bbs._sbhub_direct
_sbhub_via_address = bbs._sbhub_via_address
_enrich_with_sbhub = bbs._enrich_with_sbhub

_HEADERS = {"Authorization": "Bearer test-token", "Accept": "application/json"}


def _building(**kwargs) -> BuildingInfo:
    defaults = dict(bygningsnummer="12345678", lat=59.91, lon=10.75, distance_m=5.0)
    return BuildingInfo(**{**defaults, **kwargs})


def _resp(json_data: dict, status_code: int = 200) -> MagicMock:
    r = MagicMock()
    r.ok = status_code < 400
    r.status_code = status_code
    r.json.return_value = json_data
    return r


def _punktsok_resp(
    adressenavn: str = "Testgata",
    nummer: int = 5,
    bokstav: str | None = None,
    postnummer: str = "0001",
    poststed: str = "OSLO",
) -> MagicMock:
    return _resp(
        {
            "adresser": [
                {
                    "adressenavn": adressenavn,
                    "nummer": nummer,
                    "bokstav": bokstav,
                    "postnummer": postnummer,
                    "poststed": poststed,
                }
            ]
        }
    )


def _be_search_resp(
    units: list[tuple[int, int, str | None]],  # (bruksenhet_id, house_number, house_letter)
) -> MagicMock:
    """Build a bruksenhet-search response with house_number and house_letter per item."""
    return _resp(
        {
            "results": [
                {"bruksenhet_id": beid, "house_number": hnr, "house_letter": hlet}
                for beid, hnr, hlet in units
            ]
        }
    )


# ---------------------------------------------------------------------------
# _sbhub_direct
# ---------------------------------------------------------------------------


def test_direct_returns_enriched_building_when_api_has_data():
    building = _building()
    bygning_resp = _resp({"items": [{"bygning_id": 42}]})
    be_resp = _resp({"items": [{"bruksareal": 80.0}, {"bruksareal": 45.0}]})

    with patch.object(bbs.requests, "get", side_effect=[bygning_resp, be_resp]):
        result = _sbhub_direct(building, _HEADERS)

    assert result is not None
    assert result.bruksareal_totalt == 125.0
    assert result.antall_boenheter == 2


def test_direct_sets_bruksareal_totalt_to_none_when_all_zero():
    building = _building()
    bygning_resp = _resp({"items": [{"bygning_id": 1}]})
    be_resp = _resp({"items": [{"bruksareal": 0.0}, {"bruksareal": None}]})

    with patch.object(bbs.requests, "get", side_effect=[bygning_resp, be_resp]):
        result = _sbhub_direct(building, _HEADERS)

    assert result is not None
    assert result.bruksareal_totalt is None
    assert result.antall_boenheter == 2


# ---------------------------------------------------------------------------
# _sbhub_via_address
# ---------------------------------------------------------------------------

# Default punktsok in these tests returns nummer=5, bokstav=None (no house letter).
# The search response must use matching house_number=5, house_letter=None for filtering.

_PARCEL_ID = 100


def _address_chain(
    search_units: list[tuple[int, int, str | None]],
) -> list[MagicMock]:
    """Build the responses common to address-path tests (no matrikkel fast path).

    Default buildings have no kommunenummer, so the fast path is skipped.
    Tests that exercise the fast path must build their own response list with
    a building that has a kommunenummer in _MATRIKKEL_KOMMUNENUMMER.
    """
    return [
        _punktsok_resp(),  # nummer=5, bokstav=None
        _resp({"results": [{"matrikkelenhet_id": _PARCEL_ID}]}),
        _be_search_resp(search_units),
    ]


def test_address_returns_enriched_building_for_single_unit():
    """Happy path: one unit at this address → fetch it → return bruksareal."""
    building = _building()
    responses = [
        *_address_chain([(55, 5, None)]),  # 1 unit at nummer=5, no bokstav
        _resp({"bruksareal": 95.0, "bygg_id": 7}),  # individual fetch
    ]

    with patch.object(bbs.requests, "get", side_effect=responses):
        result = _sbhub_via_address(building, _HEADERS)

    assert result is not None
    assert result.bruksareal_totalt == 95.0
    assert result.antall_boenheter == 1


def test_address_filters_out_units_from_other_buildings_on_same_parcel():
    """Units at a different house number on the same parcel must be excluded."""
    building = _building()
    # Parcel has two buildings: our target at nummer=5 and a neighbour at nummer=7
    search_units = [
        (10, 5, None),  # our building
        (20, 5, None),  # our building
        (30, 7, None),  # neighbour — must be excluded
    ]
    responses = [
        *_address_chain(search_units),
        _resp({"bruksareal": 60.0}),  # fetch be_id=10
        _resp({"bruksareal": 70.0}),  # fetch be_id=20
    ]

    with patch.object(bbs.requests, "get", side_effect=responses):
        result = _sbhub_via_address(building, _HEADERS)

    assert result is not None
    assert result.bruksareal_totalt == 130.0
    assert result.antall_boenheter == 2


def test_address_filters_by_house_letter_when_present():
    """Buildings 5A and 5B on same parcel — only units for 5A are counted."""
    building = _building()
    search_units = [
        (10, 5, "A"),  # our building (5A)
        (20, 5, "A"),  # our building (5A)
        (30, 5, "B"),  # next-door building (5B) — must be excluded
    ]
    responses = [
        _punktsok_resp(nummer=5, bokstav="A"),  # clicked building is 5A
        _resp({"results": [{"matrikkelenhet_id": _PARCEL_ID}]}),  # address → matrikkelenhet
        _be_search_resp(search_units),
        _resp({"bruksareal": 50.0}),
        _resp({"bruksareal": 55.0}),
    ]

    with patch.object(bbs.requests, "get", side_effect=responses):
        result = _sbhub_via_address(building, _HEADERS)

    assert result is not None
    assert result.bruksareal_totalt == 105.0
    assert result.antall_boenheter == 2


_MATRIKKEL_KOMM = "1122"  # Hå — in _MATRIKKEL_KOMMUNENUMMER


def test_address_uses_matrikkel_fast_path_when_single_bygg_id():
    """Fast path fires for matrikkel municipalities; single bygg_id → no elhub fetches."""
    building = _building(kommunenummer=_MATRIKKEL_KOMM)
    responses = [
        _punktsok_resp(),
        _resp({"results": [{"matrikkelenhet_id": _PARCEL_ID}]}),
        # fast path: 2 units, same bygg_id — bruksenhet-search must NOT be called
        _resp(
            {
                "items": [
                    {"bygg_id": 9, "bruksareal": 60.0},
                    {"bygg_id": 9, "bruksareal": 40.0},
                ]
            }
        ),
    ]

    with patch.object(bbs.requests, "get", side_effect=responses):
        result = _sbhub_via_address(building, _HEADERS)

    assert result is not None
    assert result.bruksareal_totalt == 100.0
    assert result.antall_boenheter == 2


def test_address_falls_through_to_bruksenhet_search_when_multiple_bygg_ids():
    """Fast path with >1 bygg_id must fall through to bruksenhet-search."""
    building = _building(kommunenummer=_MATRIKKEL_KOMM)
    responses = [
        _punktsok_resp(),
        _resp({"results": [{"matrikkelenhet_id": _PARCEL_ID}]}),
        # fast path: different bygg_ids → can't isolate building
        _resp(
            {
                "items": [
                    {"bygg_id": 1, "bruksareal": 60.0},
                    {"bygg_id": 2, "bruksareal": 40.0},
                ]
            }
        ),
        _be_search_resp([(55, 5, None)]),
        _resp({"bruksareal": 60.0}),
    ]

    with patch.object(bbs.requests, "get", side_effect=responses):
        result = _sbhub_via_address(building, _HEADERS)

    assert result is not None
    assert result.bruksareal_totalt == 60.0


def test_address_falls_through_to_bruksenhet_search_when_fast_path_times_out():
    """Fast path timeout must fall through silently to bruksenhet-search."""
    building = _building(kommunenummer=_MATRIKKEL_KOMM)
    responses = [
        _punktsok_resp(),
        _resp({"results": [{"matrikkelenhet_id": _PARCEL_ID}]}),
        bbs.requests.exceptions.ReadTimeout("timed out"),
        _be_search_resp([(77, 5, None)]),
        _resp({"bruksareal": 95.0}),
    ]

    with patch.object(bbs.requests, "get", side_effect=responses):
        result = _sbhub_via_address(building, _HEADERS)

    assert result is not None
    assert result.bruksareal_totalt == 95.0


def test_address_deduplicates_repeated_bruksenhet_ids():
    """Duplicate bruksenhet_ids in search results must be fetched only once."""
    building = _building()
    # Same ID (77) appears three times — a known SBHub data quirk
    search_units = [(77, 5, None), (77, 5, None), (77, 5, None)]
    responses = [
        *_address_chain(search_units),
        _resp({"bruksareal": 120.0}),  # fetched exactly once despite 3 raw entries
    ]

    with patch.object(bbs.requests, "get", side_effect=responses):
        result = _sbhub_via_address(building, _HEADERS)

    assert result is not None
    assert result.bruksareal_totalt == 120.0
    assert result.antall_boenheter == 1


def test_address_sums_multiple_units_in_parallel():
    building = _building()
    search_units = [(10, 5, None), (20, 5, None), (30, 5, None)]
    responses = [
        *_address_chain(search_units),
        _resp({"bruksareal": 50.0}),
        _resp({"bruksareal": 60.0}),
        _resp({"bruksareal": 70.0}),
    ]

    with patch.object(bbs.requests, "get", side_effect=responses):
        result = _sbhub_via_address(building, _HEADERS)

    assert result is not None
    assert result.bruksareal_totalt == 180.0
    assert result.antall_boenheter == 3


# ---------------------------------------------------------------------------
# _enrich_with_sbhub (integration of both paths)
# ---------------------------------------------------------------------------


def test_enrich_returns_original_building_when_token_not_set(monkeypatch):
    monkeypatch.delenv("SBHUB_API_TOKEN", raising=False)
    building = _building()

    result = _enrich_with_sbhub(building)

    assert result is building


def test_enrich_direct_path_takes_priority(monkeypatch):
    monkeypatch.setenv("SBHUB_API_TOKEN", "test-token")
    building = _building()
    responses = [
        _resp({"items": [{"bygning_id": 5}]}),
        _resp({"items": [{"bruksareal": 120.0}]}),
    ]

    with patch.object(bbs.requests, "get", side_effect=responses):
        result = _enrich_with_sbhub(building)

    assert result.bruksareal_totalt == 120.0


def test_enrich_falls_back_to_address_path_when_direct_finds_nothing(monkeypatch):
    monkeypatch.setenv("SBHUB_API_TOKEN", "test-token")
    building = _building()
    responses = [
        _resp({"items": []}),  # direct: not in SBHub
        _punktsok_resp(),  # address path: geocode
        _resp({"results": [{"matrikkelenhet_id": 300}]}),  # address → matrikkelenhet
        _be_search_resp([(42, 5, None)]),  # bruksenhet-search
        _resp({"bruksareal": 65.0}),  # elhub individual fetch
    ]

    with patch.object(bbs.requests, "get", side_effect=responses):
        result = _enrich_with_sbhub(building)

    assert result.bruksareal_totalt == 65.0


def test_enrich_returns_original_building_when_all_calls_raise(monkeypatch):
    monkeypatch.setenv("SBHUB_API_TOKEN", "test-token")
    building = _building()

    with patch.object(bbs.requests, "get", side_effect=Exception("network error")):
        result = _enrich_with_sbhub(building)

    assert result is building
