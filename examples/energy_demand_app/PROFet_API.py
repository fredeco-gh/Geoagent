from __future__ import annotations

PROFET_BASE_URL = "https://flexibilitysuite.byggforsk.no/index.html"
PROFET_VALIDATE_URL = f"{PROFET_BASE_URL}/api/Profet/Validate"
PROFET_RUN_URL = f"{PROFET_BASE_URL}/api/Profet"


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


PROFET_EFFICIENCY_KEYS = {
    "Reg",  # Regular
    "Eff-E",  # Efficient existing / rehabilitated
    "Eff-N",  # Efficient new
    "Vef",  # Very efficient
}


def build_area_distribution(
    floor_area_m2: float,
    efficiency_key: str = "Reg",
) -> dict[str, float]:
    """
    Build an area distribution across efficiency levels.

    For a simple use case the entire floor area is placed in one level
    (typically "Reg") and the others are set to 0.
    """

    if floor_area_m2 <= 0:
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

    area_distribution[efficiency_key] = float(floor_area_m2)

    return area_distribution


def build_areas_for_one_building(
    building_category: str,
    floor_area_m2: float,
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
            floor_area_m2=floor_area_m2,
            efficiency_key=efficiency_key,
        )
    }
