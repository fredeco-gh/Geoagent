from __future__ import annotations

import dataclasses
import functools

import numpy as np
import pandas as pd
import pygfunction as gt


@dataclasses.dataclass
class BoreholeParams:
    """Geometric parameters for a single borehole."""

    H: float = 200.0  # active borehole depth [m]
    D: float = (
        0.5  # buried depth from surface to top of active section [m]  (matches Fimbul default)
    )
    r_b: float = 0.065  # borehole radius [m]
    x: float = 0.0  # horizontal position [m]
    y: float = 0.0  # horizontal position [m]
    tilt: float = 0.0  # tilt angle from vertical [radians]
    orientation: float = 0.0  # azimuthal direction of tilt [radians]


@dataclasses.dataclass
class GroundParams:
    """Ground thermal properties, uniform across the borehole field."""

    k_s: float = 3.7  # thermal conductivity [W/(m K)]  (matches Fimbul default)
    # thermal diffusivity [m²/s]  k_s/(rho*cp) = 3.7/(2580*900)  (matches Fimbul default)
    alpha: float = 1.59e-6


@dataclasses.dataclass
class PipeParams:
    """Single U-tube pipe and grout parameters."""

    r_in: float = 0.015  # inner pipe radius [m]
    r_out: float = 0.018  # outer pipe radius [m]  (r_in + 3 mm wall, matches Fimbul)
    D_s: float = (
        0.030  # shank spacing: borehole-centre to pipe-centre [m]  (pipe_spacing/2, matches Fimbul)
    )
    k_p: float = 0.38  # pipe wall thermal conductivity [W/(m K)]  (matches Fimbul)
    k_g: float = 2.3  # grout thermal conductivity [W/(m K)]  (matches Fimbul)
    epsilon: float = 1e-4  # pipe inner-wall roughness [m]  (matches Fimbul/JutulDarcy default)


@dataclasses.dataclass
class FluidParams:
    """Carrier fluid and flow parameters."""

    fluid_name: str = "Water"  # pygfunction fluid identifier
    concentration: float = 0.0  # antifreeze concentration [%] — 0 for pure water
    m_flow_per_borehole: float = 0.3  # mass flow rate per borehole [kg/s]


def rectangle_field(
    N_1: int,
    N_2: int,
    B: float = 3.0,
    H: float = 200.0,
    D: float = 0.5,
    r_b: float = 0.065,
    tilt: float = 0.0,
    orientation: float = 0.0,
    num_sectors: int | None = None,
) -> list[list[BoreholeParams]]:
    """Generate a rectangular N_1 x N_2 grid of boreholes divided into Cartesian sectors.

    Mirrors Fimbul's ``rectangular_pattern`` (uses Cartesian sector division).

    Args:
        num_sectors: Number of sectors. Defaults to N_1 * N_2 (one borehole per sector).
    """
    flat = [
        BoreholeParams(H=H, D=D, r_b=r_b, x=i * B, y=j * B, tilt=tilt, orientation=orientation)
        for i in range(N_1)
        for j in range(N_2)
    ]
    return divide_into_sectors(
        flat, num_sectors if num_sectors is not None else N_1 * N_2, "cartesian"
    )


def sunflower_field(
    N: int,
    B: float = 3.0,
    H: float = 200.0,
    D: float = 0.5,
    r_b: float = 0.065,
    tilt: float = 0.0,
    orientation: float = 0.0,
    num_sectors: int | None = None,
) -> list[list[BoreholeParams]]:
    """Generate a Fibonacci spiral pattern with N boreholes divided into angular sectors.

    Mirrors Fimbul's ``sunflower_pattern`` (uses angular sector division).

    Args:
        B: Approximate spacing between neighbouring boreholes [m].
        num_sectors: Number of sectors. Defaults to N (one borehole per sector).
    """
    phi = (1 + np.sqrt(5)) / 2
    delta_theta = 2 * np.pi / phi**2
    # Scale radius so nearest-neighbour spacing matches B (factor from Fimbul's utils.jl)
    p0 = np.array([np.sqrt(0 / N), 0.0])
    p1 = np.array([np.sqrt(1 / N) * np.cos(delta_theta), np.sqrt(1 / N) * np.sin(delta_theta)])
    current_spacing = float(np.linalg.norm(p1 - p0)) or 1.0
    radius = 1.3513 * B / current_spacing if current_spacing > 0 else B
    flat = [
        BoreholeParams(
            H=H,
            D=D,
            r_b=r_b,
            x=float(np.sqrt(i / N) * radius * np.cos(i * delta_theta)),
            y=float(np.sqrt(i / N) * radius * np.sin(i * delta_theta)),
            tilt=tilt,
            orientation=orientation,
        )
        for i in range(N)
    ]
    return divide_into_sectors(flat, num_sectors if num_sectors is not None else N, "angular")


def circular_field(
    N: int,
    B: float = 3.0,
    H: float = 200.0,
    D: float = 0.5,
    r_b: float = 0.065,
    tilt: float = 0.0,
    orientation: float = 0.0,
    num_sectors: int | None = None,
) -> list[list[BoreholeParams]]:
    """Generate N boreholes in concentric rings divided into angular sectors.

    Mirrors Fimbul's ``circular_pattern`` (uses angular sector division).
    Places one borehole at the centre, then fills rings outward until N
    boreholes are placed. Each ring holds as many boreholes as its
    circumference allows at spacing B.

    Args:
        num_sectors: Number of sectors. Defaults to N (one borehole per sector).
    """
    points: list[tuple[float, float]] = [(0.0, 0.0)]
    ring = 1
    while len(points) < N:
        r = ring * B
        n_ring = min(max(1, round(2 * np.pi * r / B)), N - len(points))
        for k in range(n_ring):
            theta = 2 * np.pi * k / n_ring
            points.append((r * np.cos(theta), r * np.sin(theta)))
        ring += 1
    flat = [
        BoreholeParams(H=H, D=D, r_b=r_b, x=x, y=y, tilt=tilt, orientation=orientation)
        for x, y in points
    ]
    return divide_into_sectors(flat, num_sectors if num_sectors is not None else N, "angular")


def _factor_pair_closest_to_square(n: int) -> tuple[int, int]:
    """Return the divisor pair (smaller, larger) of n whose product is n, closest to sqrt(n)."""
    best = (1, n)
    for d in range(1, int(n**0.5) + 1):
        if n % d == 0:
            best = (d, n // d)
    return best


def _equal_index_ranges(total: int, n_groups: int) -> list[tuple[int, int]]:
    """Split *total* into *n_groups* contiguous, as-equal-as-possible half-open [a, b) ranges."""
    base = total // n_groups
    rem = total - base * n_groups
    ranges: list[tuple[int, int]] = []
    s = 0
    for i in range(n_groups):
        sz = base + (1 if i < rem else 0)
        ranges.append((s, s + sz))
        s += sz
    return ranges


def group_into_sectors_angular(
    boreholes: list[BoreholeParams],
    num_sectors: int,
) -> list[list[BoreholeParams]]:
    """Divide a flat borehole list into *num_sectors* angular slices around the field centre.

    Mirrors ``group_into_sectors_angular`` in Fimbul/src/meshing/utils.jl.

    Each sector covers a roughly equal arc of the full 360°. Within a sector,
    boreholes are ordered by distance from the centroid (closest first), which
    defines the series-flow path that Fimbul uses.

    Args:
        boreholes: Flat list of boreholes (e.g. from ``rectangle_field``).
        num_sectors: Number of sectors to create. Must be ≥ 1 and ≤ len(boreholes).

    Returns:
        Nested list ``sectors[s][k]`` — outer index is sector, inner index is the
        position of the borehole in the series chain for that sector.
    """
    n = len(boreholes)
    if num_sectors < 1 or num_sectors > n:
        raise ValueError(f"num_sectors must be in [1, {n}]; got {num_sectors}")

    xs = np.array([b.x for b in boreholes])
    ys = np.array([b.y for b in boreholes])
    theta = np.arctan2(ys, xs) + np.pi  # [0, 2π)
    r = np.hypot(xs, ys)
    order_theta = np.argsort(theta, kind="stable")

    base = n // num_sectors
    rem = n - base * num_sectors
    sizes = [base + (1 if i < rem else 0) for i in range(num_sectors)]

    sectors: list[list[BoreholeParams]] = []
    offset = 0
    for size in sizes:
        idx = order_theta[offset : offset + size]
        idx = idx[np.argsort(r[idx], kind="stable")]  # closest-first within sector
        sectors.append([boreholes[i] for i in idx])
        offset += size
    return sectors


def group_into_sectors_cartesian(
    boreholes: list[BoreholeParams],
    num_sectors: int,
) -> list[list[BoreholeParams]]:
    """Divide a flat borehole list into *num_sectors* Cartesian grid blocks.

    Mirrors ``group_into_sectors_cartesian`` in Fimbul/src/meshing/utils.jl.

    Works best for rectangular fields where wells sit on a regular x-y grid.
    Chooses the row- and column-band counts by factoring *num_sectors* into the
    pair closest to a square (``_factor_pair_closest_to_square``), then assigns
    the larger count to the axis that has more unique coordinate values.  Within
    each block, boreholes are sorted in a boustrophedon (column-by-column zigzag)
    order that defines the series-flow path.

    Args:
        boreholes: Flat list of boreholes.
        num_sectors: Number of sectors. Must evenly factorise into a sensible
            row x col pair (e.g. 4 -> 2x2, 6 -> 2x3).

    Returns:
        Nested list ``sectors[s][k]`` ready for ``simulate_borehole_temperatures``.
    """
    n = len(boreholes)
    if num_sectors < 1 or num_sectors > n:
        raise ValueError(f"num_sectors must be in [1, {n}]; got {num_sectors}")

    xs = np.array([b.x for b in boreholes])
    ys = np.array([b.y for b in boreholes])

    cols = np.sort(np.unique(xs))
    rows = np.sort(np.unique(ys))
    n_cols_total = len(cols)
    n_rows_total = len(rows)

    n_row_bands, n_col_bands = _factor_pair_closest_to_square(num_sectors)
    if n_cols_total < n_rows_total:
        n_row_bands, n_col_bands = n_col_bands, n_row_bands

    col_ranges = _equal_index_ranges(n_cols_total, n_col_bands)
    row_ranges = _equal_index_ranges(n_rows_total, n_row_bands)

    sectors: list[list[BoreholeParams]] = []
    for c0, c1 in col_ranges:
        band_cols = cols[c0:c1]
        band_cols_set = set(band_cols)
        # 0-indexed rank within this column band (for zigzag direction)
        col_rank = {float(xc): j for j, xc in enumerate(band_cols)}
        for r0, r1 in row_ranges:
            band_rows_set = set(rows[r0:r1])
            idx = [
                i
                for i in range(n)
                if float(xs[i]) in band_cols_set and float(ys[i]) in band_rows_set
            ]
            # boustrophedon: ascending y for even col_rank, descending for odd
            idx.sort(
                key=lambda i: (
                    xs[i],
                    ys[i] if col_rank[float(xs[i])] % 2 == 0 else -ys[i],
                )
            )
            sectors.append([boreholes[i] for i in idx])
    return sectors


def divide_into_sectors(
    boreholes: list[BoreholeParams],
    num_sectors: int,
    method: str = "angular",
) -> list[list[BoreholeParams]]:
    """Divide a flat borehole list into sectors for a series-parallel network.

    Wraps ``group_into_sectors_angular`` and ``group_into_sectors_cartesian``,
    matching the ``sector_division`` dispatch in Fimbul's ``field_from_points``.

    The returned nested list is ready to pass directly to
    ``simulate_borehole_temperatures`` as the ``boreholes`` argument: each inner
    list is one sector (boreholes connected in series, in the order listed) and
    the outer list collects all sectors (connected in parallel at the network
    inlet/outlet).

    Args:
        boreholes: Flat list of boreholes from any of the field generators.
        num_sectors: Number of sectors. Pass 1 to keep all boreholes in a single
            series chain; pass ``len(boreholes)`` for a fully parallel field.
        method: ``"angular"`` (default) — equal angular slices, radially ordered
            within each slice. ``"cartesian"`` — Cartesian grid blocks with
            boustrophedon ordering; works best for rectangular fields.

    Returns:
        ``list[list[BoreholeParams]]`` with ``len(…) == num_sectors``.

    Raises:
        ValueError: for unknown *method* or invalid *num_sectors*.
    """
    if method == "angular":
        return group_into_sectors_angular(boreholes, num_sectors)
    elif method == "cartesian":
        return group_into_sectors_cartesian(boreholes, num_sectors)
    else:
        raise ValueError(
            f"Unknown sector division method {method!r}. Use 'angular' or 'cartesian'."
        )


def _points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Boolean mask: True where each column of points lies inside polygon (even-odd rule).

    points: (2, N) array of candidate coordinates.
    polygon: (2, M) array of vertex coordinates (closed implicitly).
    """
    n_pts = points.shape[1]
    n_vert = polygon.shape[1]
    result = np.zeros(n_pts, dtype=bool)
    for p in range(n_pts):
        x, y = points[0, p], points[1, p]
        inside = False
        j = n_vert - 1
        for i in range(n_vert):
            xi, yi = polygon[0, i], polygon[1, i]
            xj, yj = polygon[0, j], polygon[1, j]
            if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
                inside = not inside
            j = i
        result[p] = inside
    return result


def polygonal_field(
    N: int,
    B: float = 3.0,
    num_sides: int = 6,
    H: float = 200.0,
    D: float = 0.5,
    r_b: float = 0.065,
    tilt: float = 0.0,
    orientation: float = 0.0,
    num_sectors: int | None = None,
) -> list[list[BoreholeParams]]:
    """Generate N boreholes inside a regular polygon divided into angular sectors.

    Mirrors Fimbul's ``polygonal_pattern`` (uses angular sector division).
    Sizes the polygon to enclose approximately N boreholes at spacing B, fills
    it with a rectangular grid, keeps only interior points, then keeps the N
    closest to the centroid.

    Args:
        num_sectors: Number of sectors. Defaults to N (one borehole per sector).
    """
    R = np.sqrt(2 * N * B**2 / (num_sides * np.sin(2 * np.pi / num_sides)))
    theta = np.linspace(0, 2 * np.pi, num_sides + 1)[:-1]
    polygon = np.vstack([R * np.cos(theta), R * np.sin(theta)])  # (2, num_sides)

    margin = 1.2
    nxy = 2 * int(np.ceil(margin * R / B)) + 1
    xs = (np.arange(nxy) - (nxy - 1) / 2) * B
    xx, yy = np.meshgrid(xs, xs)
    xy = np.vstack([xx.ravel(), yy.ravel()])  # (2, nxy²)

    xy = xy[:, _points_in_polygon(xy, polygon)]
    order = np.argsort(np.hypot(xy[0], xy[1]))[:N]
    xy = xy[:, order]

    flat = [
        BoreholeParams(
            H=H,
            D=D,
            r_b=r_b,
            x=float(xy[0, i]),
            y=float(xy[1, i]),
            tilt=tilt,
            orientation=orientation,
        )
        for i in range(xy.shape[1])
    ]
    return divide_into_sectors(flat, num_sectors if num_sectors is not None else N, "angular")


@functools.lru_cache(maxsize=256)
def fetch_mean_surface_temperature(
    lat: float,
    lon: float,
    start_year: int = 1991,
    end_year: int = 2020,
) -> float:
    """Return the ERA5-Land climate normal for 2 m air temperature at a location.

    Fetches daily mean temperatures from Open-Meteo's ERA5-Land archive and
    averages them over the requested period.  Daily resolution keeps the
    request compact — one value per day instead of 24.

    Args:
        lat: Latitude [degrees N].
        lon: Longitude [degrees E].
        start_year: First year of the averaging period (inclusive).
        end_year: Last year of the averaging period (inclusive).

    Returns:
        Mean 2 m air temperature over the period [degrees C].
    """
    import requests

    resp = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": f"{start_year}-01-01",
            "end_date": f"{end_year}-12-31",
            "daily": "temperature_2m_mean",
            "timezone": "UTC",
            "models": "era5_land",
        },
        timeout=180,
    )
    resp.raise_for_status()
    daily_temps = resp.json()["daily"]["temperature_2m_mean"]
    valid = [t for t in daily_temps if t is not None]
    if not valid:
        raise ValueError(
            f"No temperature data returned for ({lat}, {lon}) between {start_year} and {end_year}."
        )
    return float(sum(valid) / len(valid))


def compute_undisturbed_ground_temperature(
    T_surface: float,
    H: float,
    D: float,
    G: float = 0.03,
    tilt: float = 0.0,
) -> float:
    """Compute the representative undisturbed ground temperature for a single borehole.

    T_g = T_surface + G * (D + cos(tilt) * H / 2)

    Args:
        T_surface: Mean annual surface temperature [degrees C].
        H: Active borehole depth [m].
        D: Buried depth from surface to top of active section [m].
        G: Geothermal gradient [K/m].
        tilt: Borehole tilt angle from vertical [radians]. 0 for vertical.

    Returns:
        Undisturbed ground temperature [degrees C].
    """
    return T_surface + G * (D + np.cos(tilt) * H / 2.0)


def compute_field_undisturbed_ground_temperature(
    T_surface: float,
    boreholes: list,
    G: float = 0.03,
) -> float:
    """Compute the length-weighted mean undisturbed ground temperature for a borehole field.

    T_g,field = sum_i(H_i * T_g,i) / sum_i(H_i)

    where T_g,i = T_surface + G * (D_i + cos(tilt_i) * H_i / 2)

    This is the formula from the PDF for fields with potentially varying borehole
    depths, buried depths, or tilt angles. For a uniform field it collapses to the
    single-borehole formula.

    Args:
        T_surface: Mean annual surface temperature [degrees C].
        boreholes: List of pygfunction Borehole objects (each has .H, .D, .tilt).
        G: Geothermal gradient [K/m].

    Returns:
        Length-weighted mean undisturbed ground temperature [degrees C].
    """
    total_H = sum(b.H for b in boreholes)
    weighted = sum(
        b.H * compute_undisturbed_ground_temperature(T_surface, b.H, b.D, G, b.tilt)
        for b in boreholes
    )
    return weighted / total_H


def boreholes_to_fimbul_field(
    boreholes: list[list[BoreholeParams]],
) -> list[list[list[list[float]]]]:
    """Convert a nested list of BoreholeParams to Fimbul's field structure.

    Returns list[sector][well] = [[x_top, y_top, z_top], [x_bot, y_bot, z_bot]],
    where the top point is (b.x, b.y, b.D) and the bottom point is displaced
    by H along the borehole axis defined by tilt and orientation.
    """
    import math

    field = []
    for sector in boreholes:
        sector_wells = []
        for b in sector:
            x_bot = b.x + b.H * math.sin(b.tilt) * math.cos(b.orientation)
            y_bot = b.y + b.H * math.sin(b.tilt) * math.sin(b.orientation)
            z_bot = b.D + b.H * math.cos(b.tilt)
            sector_wells.append([[b.x, b.y, b.D], [x_bot, y_bot, z_bot]])
        field.append(sector_wells)
    return field


def simulate_borehole_temperatures(
    df_demand: pd.DataFrame,
    lat: float | None = None,
    lon: float | None = None,
    boreholes: list[list[BoreholeParams]] | None = None,
    pipe: PipeParams | None = None,
    fluid: FluidParams | None = None,
    ground: GroundParams | None = None,
    COP: float = 3.5,
    G: float = 0.03,
    year: int | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Compute hourly fluid temperatures in a borehole field from a building thermal demand.

    Converts building heating demand to a ground extraction load via
    Q_borehole = Q_building * (COP - 1) / COP, then uses pygfunction's
    g-function / Claesson-Javed load aggregation to obtain the borehole wall
    temperature T_b and pygfunction's Network pipe model to obtain carrier-fluid
    inlet and outlet temperatures T_in / T_out at each time step.

    The field is organised as a series-parallel network: ``boreholes`` is a nested
    list where each inner list is one *sector*.  Boreholes within a sector are
    connected in series (fluid flows sequentially in the order listed); sectors
    are connected in parallel (all sectors share the same network inlet).
    Pass ``[[b] for b in flat_list]`` to get an all-parallel field (one borehole
    per sector), which recovers the standard BTES parallel configuration.

    When lat and lon are provided, the undisturbed ground temperature T_g is
    computed automatically from the ERA5-Land 1991-2020 climate normal and G.

    Args:
        df_demand:
            DataFrame with a 'time' column and one kW demand column (any name).
            The first non-'time' column is used as the demand series.
        lat: Latitude of the site [degrees N]. Used to fetch T_surface from ERA5.
        lon: Longitude of the site [degrees E].
        boreholes:
            Nested list of BoreholeParams.  Outer list = sectors (parallel);
            inner list = boreholes within a sector (series, in order).
            Defaults to a single sector containing one 200 m borehole.
        pipe:
            Single U-tube pipe and grout parameters.
            Defaults to 40 mm OD HDPE pipes in standard grout.
        fluid:
            Carrier fluid, antifreeze concentration, and mass flow rate per
            borehole (= mass flow rate per sector, since boreholes in a sector
            share the same flow).  Defaults to pure water at 0.3 kg/s.
        COP:
            Heat-pump COP used to derive the ground extraction load.
            Q_borehole = Q_building * (COP - 1) / COP.
        ground: Ground thermal properties (k_s, alpha). Uniform across the field.
        G: Geothermal gradient [K/m] used when computing T_g from lat/lon.

    Returns:
        DataFrame with columns:
            time            -- original timestamps
            Q_building_kW   -- building thermal demand [kW]
            Q_borehole_kW   -- ground extraction load [kW]
            T_in            -- network inlet temperature [degrees C]
            T_out           -- network outlet temperature [degrees C]
    """
    if boreholes is None:
        boreholes = [[BoreholeParams()]]
    if pipe is None:
        pipe = PipeParams()
    if fluid is None:
        fluid = FluidParams()
    if ground is None:
        ground = GroundParams()

    # -- demand series --
    df = df_demand.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    demand_cols = [c for c in df.columns if c != "time"]
    if not demand_cols:
        raise ValueError("df_demand must contain at least one column besides 'time'.")
    demand_col = demand_cols[0]

    Q_building_W = df[demand_col].to_numpy(dtype=float) * 1_000.0  # kW -> W
    n_hours = len(Q_building_W)
    dt = 3_600.0  # 1-hour time step [s]
    tmax = n_hours * dt

    # Ground extraction load: positive = heat drawn from the ground
    Q_borehole_W = Q_building_W * (COP - 1.0) / COP

    # -- borehole field --
    # Flatten the nested sector list; preserve sector structure for connectivity.
    boreholes_flat = [b for sector in boreholes for b in sector]
    N_sectors = len(boreholes)
    boreholes_gt = [
        gt.boreholes.Borehole(b.H, b.D, b.r_b, b.x, b.y, b.tilt, b.orientation)
        for b in boreholes_flat
    ]
    H_total = sum(b.H for b in boreholes_gt)

    # bore_connectivity[i] = index of the upstream borehole that feeds borehole i,
    # or -1 if borehole i is connected directly to the network inlet.
    bore_connectivity: list[int] = []
    offset = 0
    for sector in boreholes:
        for j in range(len(sector)):
            bore_connectivity.append(-1 if j == 0 else offset + j - 1)
        offset += len(sector)

    # -- undisturbed ground temperature --
    # Computed after the field is built so we can use each borehole's actual H, D,
    # and tilt in the length-weighted average (PDF formula for multi-borehole fields).
    if lat is not None and lon is not None:
        kw = {"start_year": year, "end_year": year} if year is not None else {}
        T_surface = fetch_mean_surface_temperature(lat, lon, **kw)
    else:
        T_surface = 7.0  # Oslo annual mean — fallback when no location is given
    T_g = compute_field_undisturbed_ground_temperature(T_surface, boreholes_gt, G)

    # -- fluid properties --
    fl = gt.media.Fluid(fluid.fluid_name, fluid.concentration)
    cp_f = fl.cp  # specific heat capacity [J/(kg K)]
    rho_f = fl.rho  # density [kg/m^3]
    mu_f = fl.mu  # dynamic viscosity [Pa s]
    k_f = fl.k  # thermal conductivity [W/(m K)]
    m_flow = fluid.m_flow_per_borehole

    # -- pipe thermal resistance: fluid film + pipe wall --
    R_p = gt.pipes.conduction_thermal_resistance_circular_pipe(pipe.r_in, pipe.r_out, pipe.k_p)
    h_f = gt.pipes.convective_heat_transfer_coefficient_circular_pipe(
        m_flow, pipe.r_in, mu_f, rho_f, k_f, cp_f, pipe.epsilon
    )
    R_fp = 1.0 / (h_f * 2.0 * np.pi * pipe.r_in) + R_p  # [(m K)/W]

    # -- series-parallel network --
    # One SingleUTube per borehole (own r_b → own grout resistance; own H → own NTU).
    pos = [(-pipe.D_s, 0.0), (pipe.D_s, 0.0)]
    pipe_objs = [
        gt.pipes.SingleUTube(pos, pipe.r_in, pipe.r_out, b, ground.k_s, pipe.k_g, R_fp)
        for b in boreholes_gt
    ]
    network = gt.networks.Network(boreholes_gt, pipe_objs, bore_connectivity)

    # Sectors are in parallel -> total network flow = N_sectors * m_flow_per_sector.
    # Boreholes within a sector are in series → same m_flow passes through each.
    m_flow_network = N_sectors * m_flow

    # -- Claesson-Javed load aggregation --
    LoadAgg = gt.load_aggregation.ClaessonJaved(dt, tmax)
    time_req = LoadAgg.get_times_for_simulation()

    # -- g-function --
    # 'equivalent' is fastest but only handles vertical boreholes; fall back to
    # 'similarities' when any borehole has a non-zero tilt.
    method = "similarities" if any(b.tilt != 0.0 for b in boreholes_gt) else "equivalent"
    # nSegments controls per-borehole axial discretisation in the g-function.
    # Accuracy gain drops quickly with field size (borehole interactions dominate),
    # while memory scales as (N*nSegments)^2. Cap at 1 for large fields.
    n_boreholes = len(boreholes_gt)
    n_segments = 8 if n_boreholes <= 10 else (4 if n_boreholes <= 30 else 1)
    gFunc = gt.gfunction.gFunction(
        boreholes_gt,
        ground.alpha,
        time=time_req,
        method=method,
        options={"nSegments": n_segments},
    )
    LoadAgg.initialize(gFunc.gFunc / (2.0 * np.pi * ground.k_s))

    # -- hourly simulation loop --
    time_s = np.arange(1, n_hours + 1) * dt  # cumulative seconds
    T_in_arr = np.empty(n_hours)
    T_out_arr = np.empty(n_hours)

    for i in range(n_hours):
        LoadAgg.next_time_step(time_s[i])
        LoadAgg.set_current_load(Q_borehole_W[i] / H_total)  # W/m
        T_b = T_g - LoadAgg.temporal_superposition()

        T_in_arr[i] = network.get_network_inlet_temperature(
            Q_borehole_W[i], T_b, m_flow_network, cp_f, nSegments=1
        )
        T_out_arr[i] = network.get_network_outlet_temperature(
            T_in_arr[i], T_b, m_flow_network, cp_f, nSegments=1
        )

    df_out = pd.DataFrame(
        {
            "time": df["time"].values,
            "Q_building_kW": Q_building_W / 1_000.0,
            "Q_borehole_kW": Q_borehole_W / 1_000.0,
            "T_in": T_in_arr,
            "T_out": T_out_arr,
        }
    )
    return df_out, time_req, gFunc.gFunc
