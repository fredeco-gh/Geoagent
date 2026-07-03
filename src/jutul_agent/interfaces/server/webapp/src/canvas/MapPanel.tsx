// The geothermal map as a native canvas panel — see docs/web-ui.md's
// "Extending the canvas". Ported from geothermal-viz's web/js/app.js
// (rendering: layers, popups, well info, terrain/3D buildings) onto the
// generic `ui`/`ui_event` wire Phase 0 proved (canvas/registry.tsx's
// useUiActions/onUiEvent): a tool drives the map by sending a `ui` action
// targeted at this view (see examples/geothermal-map/capability.py), and a
// well click is relayed back as a `ui_event`, the same way geothermal-viz's
// jutul-agent-bridge.js used to relay it over postMessage.
//
// Borehole data is fetched directly from a static mount (examples/geothermal-map
// /serve.py's `extra_mounts`) rather than through any session/Julia kernel —
// a missing or broken mount must not break the rest of the app, so a failed
// fetch just leaves the map without well markers (see the principle in
// docs/web-ui.md).

import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { useCallback, useEffect, useRef, useState } from "react";

import "./MapPanel.css";
import type { PanelProps } from "./registry";
import { useUiActions } from "./registry";
import { createWellbore3D, type Wellbore3D } from "./wellbore3d";

interface GeoJsonFeature {
  type: "Feature";
  geometry: { type: string; coordinates: [number, number] };
  properties: Record<string, unknown>;
}

interface GeoJsonFeatureCollection {
  type: "FeatureCollection";
  features: GeoJsonFeature[];
}

const DATA_URL = "/geothermal-data/all_boreholes.geojson";

const MAP_CENTER: [number, number] = [10.75, 59.91]; // Oslo
const MAP_ZOOM = 11;
const MAP_PITCH = 45;
const MAP_BEARING = -10;

interface LayerConfig {
  id: string;
  color: string;
  label: string;
}

// Norwegian source layer name -> the group it renders as (several minor
// layers share the "other" group's source/layer/checkbox).
const LAYER_CONFIG: Record<string, LayerConfig> = {
  EnergiBrønn: { id: "energibronn", color: "#e74c3c", label: "Energy Well" },
  GrunnvannBrønn: { id: "grunnvannbronn", color: "#3498db", label: "Groundwater Well" },
  BrønnPark: { id: "bronnpark", color: "#2ecc71", label: "Well Park" },
  Sonderboring: { id: "sonderboring", color: "#f39c12", label: "Probe Drilling" },
  LGNBrønn: { id: "other", color: "#9b59b6", label: "LGN Well" },
  GrunnvannOppkomme: { id: "other", color: "#9b59b6", label: "Spring" },
  LGNOmrådeRefPkt: { id: "other", color: "#9b59b6", label: "LGN Ref. Point" },
};

// One toggle per group shown in the sidebar.
const LAYER_GROUPS: LayerConfig[] = [
  { id: "energibronn", color: "#e74c3c", label: "Energy Wells (EnergiBrønn)" },
  { id: "grunnvannbronn", color: "#3498db", label: "Groundwater Wells (GrunnvannBrønn)" },
  { id: "bronnpark", color: "#2ecc71", label: "Well Parks (BrønnPark)" },
  { id: "sonderboring", color: "#f39c12", label: "Probe Drillings (Sonderboring)" },
  { id: "other", color: "#9b59b6", label: "Other (LGN, Springs)" },
];

// Norwegian -> English field labels for the well-info table.
const FIELD_LABELS: Record<string, string> = {
  brønnNr: "Well No.",
  objekttype: "Type",
  boretLengde: "Drilled Length (m)",
  boretLengdeTilBerg: "Depth to Bedrock (m)",
  boreDato: "Drill Date",
  diameterBorehull: "Borehole Diameter (mm)",
  vannstandBorehull: "Water Level (m)",
  boretKapasitet: "Capacity (l/h)",
  materialForingsrør: "Casing Material",
  lengdeForingsrør: "Casing Length (m)",
  brønnHelningType: "Inclination Type",
  boretHelningsgrad: "Inclination (°)",
  boretAzimuth: "Azimuth (°)",
  oppdragstaker: "Contractor",
  konsulentFirma: "Consultant",
  beskrivelse: "Description",
  brønnParkNr: "Well Park No.",
  brønnpOmrNavn: "Area Name",
  antallEnergiBrønner: "No. of Energy Wells",
  brønnpVEffekt: "Heating Power (kW)",
  brønnpVEnergi: "Heating Energy (MWh)",
  brønnpKEffekt: "Cooling Power (kW)",
  brønnpKEnergi: "Cooling Energy (MWh)",
  brønnpFrikjøling: "Free Cooling",
  brønnpKollVæske: "Collector Fluid",
  geolMedium: "Geological Medium",
};

interface SelectedBuilding {
  bygningsnummer: string;
  bygningstype?: string;
  bygningsstatus?: string;
  kommunenummer?: string;
  kommunenavn?: string;
  lat: number;
  lon: number;
  distance_m: number;
  bruksareal_totalt?: number;
  bruksareal_til_bolig?: number;
  bruksareal_til_annet?: number;
  antall_boenheter?: number;
}

interface BuildingClickResult {
  hit: boolean;
  selected?: SelectedBuilding;
}

interface TemperatureRecord {
  time_oslo: string;
  temperature_C: number;
}

interface BuildingTemperatureData {
  lat: number;
  lon: number;
  year: number;
  hours: number;
  records: TemperatureRecord[];
}

interface SelectedWell {
  title: string;
  color: string;
  label: string;
  rows: Array<[string, string]>;
  properties: Record<string, unknown>;
  layer: string;
  lngLat: { lng: number; lat: number };
}

// Well types Fimbul can simulate — matches simulation.jl's own
// SIMULATABLE_LAYERS, which is what actually decides whether
// `setup_simulation` returns `simulatable: true`. Kept here too so the
// "Setup Simulation" button only appears when it would actually do something.
const SIMULATABLE_LAYERS = new Set(["EnergiBrønn", "BrønnPark"]);

interface SimParamMeta {
  label: string;
  unit: string;
  min?: number;
  max?: number;
  step?: number;
  tooltip?: string;
  group?: string;
}

interface SimSetup {
  simulatable: boolean;
  case_type: string | null;
  case_label?: string;
  case_description?: string;
  well_id: string;
  parameters: Record<string, number>;
  parameter_order: string[];
  metadata: Record<string, SimParamMeta>;
  sources: Record<string, string>;
}

interface SimStatus {
  kind: "running" | "error";
  message: string;
}

function groupSimParams(setup: SimSetup): Array<[string, Array<{ key: string; meta: SimParamMeta }>]> {
  const groups = new Map<string, Array<{ key: string; meta: SimParamMeta }>>();
  for (const key of setup.parameter_order) {
    const meta = setup.metadata[key] ?? { label: key, unit: "" };
    const group = meta.group || "Other";
    if (!groups.has(group)) groups.set(group, []);
    groups.get(group)!.push({ key, meta });
  }
  return Array.from(groups.entries());
}

function escapeHtml(value: unknown): string {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function formatDate(value: unknown): string {
  if (!value) return "";
  const d = new Date(String(value));
  if (Number.isNaN(d.getTime())) return String(value);
  return d.toLocaleDateString("en-GB", { year: "numeric", month: "short", day: "numeric" });
}

function polygonCentroid(rings: [number, number][][]): { lng: number; lat: number } {
  const ring = rings[0];
  let lng = 0, lat = 0;
  for (const [x, y] of ring) { lng += x; lat += y; }
  return { lng: lng / ring.length, lat: lat / ring.length };
}

function pointInRing(px: number, py: number, ring: [number, number][]): boolean {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const xi = ring[i][0], yi = ring[i][1];
    const xj = ring[j][0], yj = ring[j][1];
    if (yi > py !== yj > py && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi) {
      inside = !inside;
    }
  }
  return inside;
}

// When a hover feature is a MultiPolygon (adjacent buildings merged into one
// OSM way), extract only the sub-polygon the cursor actually sits inside.
function hoverGeometry(
  geom: { type: string; coordinates: unknown },
  lngLat: { lng: number; lat: number },
): { type: string; coordinates: unknown } {
  const [px, py] = [lngLat.lng, lngLat.lat];
  if (geom.type === "MultiPolygon") {
    const polys = geom.coordinates as [number, number][][][];
    for (const poly of polys) {
      if (pointInRing(px, py, poly[0])) {
        return { type: "Polygon", coordinates: poly };
      }
    }
  }
  return geom;
}

function describeFeature(props: Record<string, unknown>): { title: string; color: string; label: string } {
  const layerName = String(props.layer || "Unknown");
  const cfg = LAYER_CONFIG[layerName] ?? { id: "other", color: "#999", label: layerName };
  const title = props.brønnNr
    ? `Well #${props.brønnNr}`
    : props.brønnParkNr
      ? `Well Park #${props.brønnParkNr}`
      : cfg.label;
  return { title, color: cfg.color, label: cfg.label };
}

const MAP_STYLE: maplibregl.StyleSpecification = {
  version: 8,
  sources: {
    "osm-raster": {
      type: "raster",
      tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
      tileSize: 256,
      attribution: '&copy; OpenStreetMap contributors',
    },
  },
  layers: [{ id: "osm-base", type: "raster", source: "osm-raster", minzoom: 0, maxzoom: 19 }],
};

export function MapPanel({ view, active, reloadToken, onLoaded, onUiEvent, onAction }: PanelProps) {
  const panelRef = useRef<HTMLDivElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const popupRef = useRef<maplibregl.Popup | null>(null);
  const wellbore3dRef = useRef<Wellbore3D | null>(null);
  const selectWellRef = useRef<(feature: GeoJsonFeature, lngLat: { lng: number; lat: number }) => void>(
    () => {},
  );
  const uiActions = useUiActions(view.id);

  const [selected, setSelected] = useState<SelectedWell | null>(null);
  const [selectedBuilding, setSelectedBuilding] = useState<SelectedBuilding | null>(null);
  const buildingTemperatureRef = useRef<BuildingTemperatureData | null>(null);
  // Mirrors `selected` for the ui-actions effect below, which must read the
  // currently selected well's lngLat without depending on (and re-running
  // for) every selection change.
  const selectedRef = useRef<SelectedWell | null>(null);
  selectedRef.current = selected;
  const [total, setTotal] = useState(0);
  const [byGroup, setByGroup] = useState<Record<string, number>>({});
  const [visibility, setVisibility] = useState<Record<string, boolean>>(() =>
    Object.fromEntries(LAYER_GROUPS.map((g) => [g.id, true])),
  );
  const [collapsed, setCollapsed] = useState(false);
  const [sidebarWidth, setSidebarWidth] = useState(320);

  const handleResizeMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    const panelLeft = panelRef.current?.getBoundingClientRect().left ?? 0;
    const onMove = (ev: MouseEvent) => {
      setSidebarWidth(Math.max(220, Math.min(600, ev.clientX - panelLeft)));
    };
    const onUp = () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  }, []);

  // Setup Simulation sidebar panel: a well's resolved parameters arrive as a
  // targeted `simulation_params` ui action (see capability.py's
  // make_setup_simulation_action); "Run" then fires `run_simulation` as a
  // direct action too — the agent never has to interpret either request.
  const [simPanelOpen, setSimPanelOpen] = useState(false);
  const [simSetup, setSimSetup] = useState<SimSetup | null>(null);
  const [simParams, setSimParams] = useState<Record<string, number>>({});
  const [simError, setSimError] = useState<string | null>(null);
  const [simStatus, setSimStatus] = useState<SimStatus | null>(null);

  const [energyPanelOpen, setEnergyPanelOpen] = useState(false);
  const [energyYear, setEnergyYear] = useState(2023);
  const [energyStatus, setEnergyStatus] = useState<SimStatus | null>(null);

  // A ref (not a plain function) so the mount effect below — which only runs
  // once per view.id — always calls the latest version, without needing to
  // depend on (and re-run for) every state setter it closes over.
  selectWellRef.current = (feature, lngLat) => {
    const props = feature.properties;
    const { title, color, label } = describeFeature(props);
    const layerName = String(props.layer || "Unknown");

    setSelectedBuilding(null);
    buildingTemperatureRef.current = null;
    setEnergyPanelOpen(false);
    wellbore3dRef.current?.remove();
    popupRef.current?.remove();
    const map = mapRef.current;
    if (map) {
      popupRef.current = new maplibregl.Popup({ offset: [0, 10], anchor: "top", maxWidth: "300px" })
        .setLngLat(lngLat)
        .setHTML(
          `<div class="popup-title">${escapeHtml(title)}</div>` +
            `<span class="popup-type" style="background:${color}">${escapeHtml(label)}</span>` +
            `<div class="popup-detail">` +
            (props.boretLengde ? `Depth: ${escapeHtml(props.boretLengde)} m<br>` : "") +
            (props.boreDato ? `Drilled: ${escapeHtml(formatDate(props.boreDato))}<br>` : "") +
            (props.oppdragstaker ? `Contractor: ${escapeHtml(props.oppdragstaker)}` : "") +
            `</div>`,
        )
        .addTo(map);
    }

    const rows: Array<[string, string]> = [];
    for (const [key, value] of Object.entries(props)) {
      if (key === "layer" || value === null || value === undefined || value === "") continue;
      const fieldLabel = FIELD_LABELS[key] || key;
      const display =
        key.toLowerCase().includes("dato") || key.toLowerCase().includes("date")
          ? formatDate(value)
          : String(value);
      rows.push([fieldLabel, display]);
    }
    setSelected({ title, color, label, rows, properties: props, layer: layerName, lngLat });

    // Relay the selection to the agent — mirrors geothermal-viz's own
    // wellSelected event (jutul-agent-bridge.js used to forward it over
    // postMessage); now it's a plain ui_event over the session's own socket.
    onUiEvent({ event: "wellSelected", properties: props, lngLat });
  };

  // "Back to this view" bumps `reloadToken` to force a stuck/stale panel to
  // reload — for an iframe'd panel that's a real page reload (a fresh `src`),
  // whose own `onLoad` clears the spinner naturally. The map is a
  // persistently-mounted native panel with nothing to re-fetch on "reload",
  // so without this the canvas's spinner would wait forever for an `onLoaded`
  // that was never coming. Skip the mount-time call: the real first-load
  // signal is `map.on("load", ...)` below, which this would otherwise race.
  const mountedReload = useRef(false);
  useEffect(() => {
    if (!mountedReload.current) {
      mountedReload.current = true;
      return;
    }
    // The map never "navigates away" the way an iframe'd report can, so there
    // is nothing to reload — instead this resets the same things going back
    // to a freshly-pinned view would give you: default camera, no selection.
    wellbore3dRef.current?.remove();
    popupRef.current?.remove();
    popupRef.current = null;
    setSelected(null);
    setSelectedBuilding(null);
    buildingTemperatureRef.current = null;
    setEnergyPanelOpen(false);
    mapRef.current?.flyTo({
      center: MAP_CENTER,
      zoom: MAP_ZOOM,
      pitch: MAP_PITCH,
      bearing: MAP_BEARING,
    });
    onLoaded();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reloadToken]);

  useEffect(() => {
    if (!containerRef.current) return;
    const map = new maplibregl.Map({
      container: containerRef.current,
      style: MAP_STYLE,
      center: MAP_CENTER,
      zoom: MAP_ZOOM,
      pitch: MAP_PITCH,
      bearing: MAP_BEARING,
      maxZoom: 18,
      minZoom: 8,
      maxPitch: 70,
    });
    map.addControl(new maplibregl.NavigationControl(), "top-right");
    map.addControl(new maplibregl.ScaleControl({ maxWidth: 200 }), "bottom-right");
    mapRef.current = map;
    wellbore3dRef.current = createWellbore3D(map);

    let loaded = false;
    const finishLoading = () => {
      if (loaded) return;
      loaded = true;
      onLoaded();
    };
    // Tied to the style finishing (not the data fetch below), so a slow or
    // broken data fetch can't strand the canvas's loading spinner.
    map.on("load", finishLoading);

    map.once("style.load", () => {
      fetch(DATA_URL)
        .then((resp) => {
          if (!resp.ok) throw new Error(`status ${resp.status}`);
          return resp.json() as Promise<GeoJsonFeatureCollection>;
        })
        .then((geojson) => {
          const groups: Record<string, { features: GeoJsonFeature[]; color: string }> = {};
          for (const group of LAYER_GROUPS) groups[group.id] = { features: [], color: group.color };
          for (const feature of geojson.features) {
            const layerName = String(feature.properties.layer || "unknown");
            const cfg = LAYER_CONFIG[layerName];
            if (cfg && groups[cfg.id]) groups[cfg.id].features.push(feature);
          }

          for (const [groupId, group] of Object.entries(groups)) {
            const sourceId = `source-${groupId}`;
            const layerId = `layer-${groupId}`;
            map.addSource(sourceId, {
              type: "geojson",
              data: { type: "FeatureCollection", features: group.features },
            });
            map.addLayer({
              id: layerId,
              type: "circle",
              source: sourceId,
              paint: {
                "circle-radius": ["interpolate", ["linear"], ["zoom"], 8, 2, 12, 5, 16, 10],
                "circle-color": group.color,
                "circle-opacity": 0.8,
                "circle-stroke-width": 1,
                "circle-stroke-color": "#ffffff",
              },
            });
            map.on("mouseenter", layerId, () => {
              map.getCanvas().style.cursor = "pointer";
            });
            map.on("mouseleave", layerId, () => {
              map.getCanvas().style.cursor = "";
            });
            map.on("click", layerId, (e) => {
              const feature = e.features?.[0];
              if (feature) selectWellRef.current(feature as unknown as GeoJsonFeature, e.lngLat);
            });
          }

          setTotal(geojson.features.length);
          setByGroup(Object.fromEntries(Object.entries(groups).map(([id, g]) => [id, g.features.length])));

          // Terrain/3D buildings are visual polish on free, keyless public
          // tile sources — best-effort, since the map is fully usable without
          // them if a source ever changes shape or goes away.
          try {
            map.addSource("terrain-source", {
              type: "raster-dem",
              tiles: ["https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"],
              tileSize: 256,
              encoding: "terrarium",
            });
            map.setTerrain({ source: "terrain-source", exaggeration: 1.5 });
          } catch (err) {
            console.warn("Could not enable terrain:", err);
          }
          try {
            // promoteId uses the OSM id property so setFeatureState targets
            // exactly one building at a time instead of all with the same tile ID.
            map.addSource("openmaptiles", {
              type: "vector",
              url: "https://tiles.openfreemap.org/planet",
              promoteId: { building: "id" },
            });
            map.addLayer({
              id: "building-footprints",
              source: "openmaptiles",
              "source-layer": "building",
              type: "fill",
              minzoom: 15,
              paint: {
                "fill-color": "rgba(0,0,0,0)",
                "fill-opacity": 0,
                "fill-outline-color": "#4a90d9",
              },
            });
            map.addSource("hover-building", {
              type: "geojson",
              data: { type: "FeatureCollection", features: [] },
            });
            map.addLayer({
              id: "building-hover-fill",
              source: "hover-building",
              type: "fill",
              paint: { "fill-color": "#60a5fa", "fill-opacity": 0.7, "fill-outline-color": "#2563eb" },
            });
            map.on("mousemove", (e) => {
              if (map.getZoom() < 15) return;
              const hoverSource = map.getSource("hover-building") as maplibregl.GeoJSONSource | undefined;
              if (!hoverSource) return;
              const features = map.queryRenderedFeatures(e.point, { layers: ["building-footprints"] });
              if (features.length > 0) {
                map.getCanvas().style.cursor = "pointer";
                const geom = hoverGeometry(features[0].geometry as { type: string; coordinates: unknown }, e.lngLat);
                hoverSource.setData({
                  type: "FeatureCollection",
                  features: [{ type: "Feature" as const, geometry: geom as unknown, properties: {} }],
                });
              } else {
                map.getCanvas().style.cursor = "";
                hoverSource.setData({ type: "FeatureCollection", features: [] });
              }
            });
            map.addLayer({
              id: "3d-buildings",
              source: "openmaptiles",
              "source-layer": "building",
              type: "fill-extrusion",
              minzoom: 14,
              paint: {
                "fill-extrusion-color": "#ddd",
                "fill-extrusion-height": ["get", "render_height"],
                "fill-extrusion-base": ["get", "render_min_height"],
                "fill-extrusion-opacity": 0.6,
              },
            });
          } catch (err) {
            console.warn("Could not add 3D buildings:", err);
          }
        })
        .catch((err) => {
          // A missing/broken data mount must not break the rest of the app —
          // the map still renders, just with no well markers.
          console.warn("Could not load borehole data:", err);
        })
        .finally(finishLoading);

      // Building click: only at zoom >= 15, yields to well point clicks.
      // Uses the clicked polygon's centroid for the Matrikkelen WFS lookup so
      // any click inside the footprint works, not just near the registered point.
      // Silently no-ops when the backend endpoint is not available.
      const wellLayerIds = LAYER_GROUPS.map((g) => `layer-${g.id}`);
      map.on("click", async (e) => {
        if (map.getZoom() < 15) return;
        if (map.queryRenderedFeatures(e.point, { layers: wellLayerIds }).length > 0) return;
        // Prefer polygon centroid for accuracy; fall back to raw click coordinates
        // when building-footprints tiles haven't loaded yet.
        let lat = e.lngLat.lat;
        let lon = e.lngLat.lng;
        const buildingFeatures = map.queryRenderedFeatures(e.point, { layers: ["building-footprints"] });
        if (buildingFeatures.length === 0) return;
        const geom = hoverGeometry(
          buildingFeatures[0].geometry as { type: string; coordinates: unknown },
          e.lngLat,
        );
        if (geom.type === "Polygon" || geom.type === "MultiPolygon") {
          const coords = geom.coordinates as [number, number][][] | [number, number][][][];
          const rings = (geom.type === "Polygon" ? coords : (coords as [number, number][][][])[0]) as [number, number][][];
          const centroid = polygonCentroid(rings);
          lat = centroid.lat;
          lon = centroid.lng;
        }
        popupRef.current?.remove();
        setSelected(null);
        buildingTemperatureRef.current = null;
        setSimPanelOpen(false);

        const startTemperatureFetch = (bLat: number, bLon: number) => {
          fetch(`/api/building-temperature?lat=${bLat}&lon=${bLon}&year=2023`)
            .then((r) => (r.ok ? r.json() : null))
            .then((data) => {
              if (data) {
                buildingTemperatureRef.current = data as BuildingTemperatureData;
                console.log(
                  `[Open-Meteo] Temperature saved: ${(data as BuildingTemperatureData).hours} hours for year ${(data as BuildingTemperatureData).year}`,
                );
              }
            })
            .catch(() => { /* temperature fetch is best-effort */ });
        };

        // Fetch Matrikkelen metadata for the building. The registered
        // Matrikkelen coordinates are used for the temperature lookup (primary path).
        // Temporary fallback: when the WFS is unavailable, use the polygon
        // centroid so temperature data can still be fetched.
        const res = await fetch(`/api/building-click?lat=${lat}&lon=${lon}`);
        if (!res.ok) {
          setSelectedBuilding({ bygningsnummer: "?", lat, lon, distance_m: 0 });
          setCollapsed(false);
          startTemperatureFetch(lat, lon); // temporary: centroid fallback
          return;
        }
        const result = (await res.json()) as BuildingClickResult;
        if (!result.hit || !result.selected) return;
        setSelectedBuilding(result.selected);
        setCollapsed(false);
        startTemperatureFetch(result.selected.lat, result.selected.lon);
      });
    });

    return () => {
      map.remove();
      mapRef.current = null;
      popupRef.current = null;
      wellbore3dRef.current = null;
    };
    // Mounted once for this view's lifetime; onLoaded/onUiEvent are
    // Canvas-supplied closures stable enough for the panel's own life, and
    // selectWellRef always points at the current closure (see above).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [view.id]);

  useEffect(() => {
    if (active) mapRef.current?.resize();
  }, [active]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    for (const { action, payload } of uiActions) {
      if (action === "set_map_view") {
        const { lon, lat, zoom } = payload as { lon?: number; lat?: number; zoom?: number };
        if (typeof lon === "number" && typeof lat === "number") {
          map.flyTo({ center: [lon, lat], zoom: zoom ?? map.getZoom() });
        }
      } else if (action === "go_to_well") {
        const { lon, lat, feature } = payload as {
          lon?: number;
          lat?: number;
          feature?: GeoJsonFeature;
        };
        if (typeof lon === "number" && typeof lat === "number") {
          map.flyTo({ center: [lon, lat], zoom: 17 });
          if (feature) selectWellRef.current(feature, { lng: lon, lat });
        }
      } else if (action === "simulation_params") {
        const setup = payload as unknown as SimSetup;
        setSimStatus(null);
        if (setup.simulatable) {
          setSimError(null);
          setSimSetup(setup);
          setSimParams({ ...setup.parameters });
          // Mirrors geothermal-viz's own simulationSetup event: show the 3D
          // wellbore for the well the resolved params belong to.
          const lngLat = selectedRef.current?.lngLat;
          if (lngLat) wellbore3dRef.current?.show(lngLat, setup.parameters, setup.case_type);
        } else {
          setSimSetup(null);
          setSimError("This well type does not support simulation.");
        }
      } else if (action === "simulation_setup_error") {
        const { message } = payload as { message?: string };
        setSimStatus(null);
        setSimSetup(null);
        setSimError(message || "Could not resolve simulation parameters.");
      }
    }
  }, [uiActions]);

  const handleSetupSimulation = () => {
    if (!selected) return;
    setSimPanelOpen(true);
    setSimError(null);
    setSimSetup(null);
    setSimStatus(null);
    onAction("setup_simulation", selected.properties);
  };

  const handleCloseSimPanel = () => {
    setSimPanelOpen(false);
    wellbore3dRef.current?.remove();
  };

  const handleParamChange = (key: string, raw: string) => {
    const value = parseFloat(raw);
    if (Number.isNaN(value)) return;
    setSimParams((p) => ({ ...p, [key]: value }));
    wellbore3dRef.current?.update({ [key]: value });
  };

  const handleRunSimulation = () => {
    if (!simSetup?.case_type) return;
    setSimStatus({
      kind: "running",
      message: "Sent to the agent — watch the chat for progress and results.",
    });
    onAction("run_simulation", { case_type: simSetup.case_type, parameters: simParams });
  };

  const toggleLayer = (groupId: string, checked: boolean) => {
    setVisibility((v) => ({ ...v, [groupId]: checked }));
    const map = mapRef.current;
    const layerId = `layer-${groupId}`;
    if (map?.getLayer(layerId)) {
      map.setLayoutProperty(layerId, "visibility", checked ? "visible" : "none");
    }
  };

  const visibleCount = LAYER_GROUPS.reduce(
    (sum, g) => sum + (visibility[g.id] ? byGroup[g.id] ?? 0 : 0),
    0,
  );

  return (
    <div ref={panelRef} className={`map-panel${active ? " active" : ""}`}>
      <div ref={containerRef} className="map-panel-map" />
      <div
        className={`sidebar${collapsed ? " collapsed" : ""}`}
        style={collapsed ? undefined : { width: sidebarWidth }}
      >
        <div className="sidebar-header">
          <h1>Geothermal map</h1>
          <p className="subtitle">Norwegian boreholes</p>
        </div>
        <div className="sidebar-section">
          <h2>Layers</h2>
          {LAYER_GROUPS.map((g) => (
            <label className="layer-toggle" key={g.id}>
              <input
                type="checkbox"
                checked={visibility[g.id] ?? true}
                onChange={(e) => toggleLayer(g.id, e.target.checked)}
              />
              <span className="color-dot" style={{ background: g.color }} />
              {g.label}
            </label>
          ))}
        </div>
        <div className="sidebar-section">
          <h2>Statistics</h2>
          <div className="stat-item">
            <span className="stat-label">Total boreholes:</span>
            <span className="stat-value">{total.toLocaleString()}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Visible:</span>
            <span className="stat-value">{visibleCount.toLocaleString()}</span>
          </div>
        </div>
        <div className="sidebar-section">
          {selected ? (
            <>
              <h2>Selected Well</h2>
              <div className="well-detail">
                <div className="well-title">
                  <span className="popup-type" style={{ background: selected.color }}>
                    {selected.label}
                  </span>{" "}
                  {selected.title}
                </div>
                {SIMULATABLE_LAYERS.has(selected.layer) ? (
                  <div className="sim-setup-action">
                    <button className="btn-primary" onClick={handleSetupSimulation}>
                      ⚡ Setup Simulation
                    </button>
                  </div>
                ) : null}
                <table>
                  <tbody>
                    {selected.rows.map(([label, value]) => (
                      <tr key={label}>
                        <td>{label}</td>
                        <td>{value}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          ) : selectedBuilding ? (
            <>
              <h2>Selected Building</h2>
              <div className="well-detail">
                <div className="well-title">Building {escapeHtml(selectedBuilding.bygningsnummer)}</div>
                <div className="sim-setup-action">
                  <button className="btn-primary" onClick={() => setEnergyPanelOpen(true)}>
                    ⚡ Analyze energy needs
                  </button>
                </div>
                <table>
                  <tbody>
                    <tr><td>Bygningsnummer</td><td>{escapeHtml(selectedBuilding.bygningsnummer)}</td></tr>
                    <tr><td>Bygningstype</td><td>{selectedBuilding.bygningstype ? escapeHtml(selectedBuilding.bygningstype) : "?"}</td></tr>
                    <tr><td>Bygningsstatus</td><td>{selectedBuilding.bygningsstatus ? escapeHtml(selectedBuilding.bygningsstatus) : "?"}</td></tr>
                    <tr><td>Kommunenavn</td><td>{selectedBuilding.kommunenavn ? escapeHtml(selectedBuilding.kommunenavn) : "?"}</td></tr>
                    <tr><td>Kommunenummer</td><td>{selectedBuilding.kommunenummer ? escapeHtml(selectedBuilding.kommunenummer) : "?"}</td></tr>
                    <tr><td>Latitude</td><td>{selectedBuilding.lat.toFixed(6)}</td></tr>
                    <tr><td>Longitude</td><td>{selectedBuilding.lon.toFixed(6)}</td></tr>
                    <tr><td>Distance</td><td>{selectedBuilding.distance_m.toFixed(1)} m</td></tr>
                    <tr><td>Bruksareal totalt</td><td>{selectedBuilding.bruksareal_totalt != null ? `${selectedBuilding.bruksareal_totalt} m²` : "?"}</td></tr>
                    <tr><td>Bruksareal bolig</td><td>{selectedBuilding.bruksareal_til_bolig != null ? `${selectedBuilding.bruksareal_til_bolig} m²` : "?"}</td></tr>
                    <tr><td>Bruksareal annet</td><td>{selectedBuilding.bruksareal_til_annet != null ? `${selectedBuilding.bruksareal_til_annet} m²` : "?"}</td></tr>
                    <tr><td>Antall boenheter</td><td>{selectedBuilding.antall_boenheter != null ? selectedBuilding.antall_boenheter : "?"}</td></tr>
                  </tbody>
                </table>
              </div>
            </>
          ) : (
            <>
              <h2>Selection</h2>
              <p className="no-selection">Click a well or building on the map.</p>
            </>
          )}
        </div>
      </div>
      {!collapsed && (
        <div
          className="sidebar-resize-handle"
          style={{ left: sidebarWidth - 3 }}
          onMouseDown={handleResizeMouseDown}
        />
      )}
      <button
        className={`sidebar-toggle${collapsed ? " shifted" : ""}`}
        style={collapsed ? undefined : { left: sidebarWidth + 10 }}
        title="Toggle sidebar"
        onClick={() => setCollapsed((c) => !c)}
      >
        ☰
      </button>
      <div className={`sim-panel${simPanelOpen ? " open" : ""}`}>
        <div className="sim-panel-header">
          <h2>{simSetup ? `Simulation — ${simSetup.well_id}` : "Simulation Setup"}</h2>
          <button className="btn-icon" title="Close" onClick={handleCloseSimPanel}>
            ✕
          </button>
        </div>
        <div className="sim-tab-content active">
          {simError ? (
            <p className="sim-case-desc">{simError}</p>
          ) : simSetup ? (
            <>
              <div className="sim-case-info">
                <div className="sim-well-id">{simSetup.well_id}</div>
                <div className="sim-case-badge">{simSetup.case_label}</div>
                <p className="sim-case-desc">{simSetup.case_description}</p>
              </div>
              <div className="sim-params">
                {groupSimParams(simSetup).map(([groupName, items]) => (
                  <div className="sim-param-group" key={groupName}>
                    <h3>{groupName}</h3>
                    {items.map(({ key, meta }) => {
                      const source = simSetup.sources[key];
                      const sourceClass = source === "data" ? "source-data" : "source-default";
                      const sourceLabel = source === "data" ? "from well data" : "default";
                      return (
                        <div className="sim-param-row" key={key}>
                          <label htmlFor={`sim-p-${key}`}>
                            {meta.label} <span className="sim-param-unit">{meta.unit}</span>
                          </label>
                          <div className="sim-param-input-wrap">
                            <input
                              type="number"
                              id={`sim-p-${key}`}
                              className="sim-param-input"
                              value={simParams[key] ?? ""}
                              min={meta.min}
                              max={meta.max}
                              step={meta.step}
                              onChange={(e) => handleParamChange(key, e.target.value)}
                            />
                            <span className={`sim-param-source ${sourceClass}`} title={sourceLabel}>
                              {source === "data" ? "📊" : "⚙️"}
                            </span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                ))}
              </div>
              <div className="sim-actions">
                <button
                  className="btn-primary btn-run"
                  disabled={simStatus?.kind === "running"}
                  onClick={handleRunSimulation}
                >
                  ▶ Run Simulation
                </button>
              </div>
              {simStatus ? (
                <div className={`sim-status ${simStatus.kind}`}>{simStatus.message}</div>
              ) : null}
            </>
          ) : (
            <p className="sim-case-desc">Loading simulation setup…</p>
          )}
        </div>
      </div>
      <div className={`sim-panel${energyPanelOpen ? " open" : ""}`}>
        <div className="sim-panel-header">
          <h2>Energy needs — building #{selectedBuilding?.bygningsnummer ?? "?"}</h2>
          <button className="btn-icon" title="Close" onClick={() => setEnergyPanelOpen(false)}>
            ✕
          </button>
        </div>
        <div className="sim-tab-content active">
          <div className="sim-param-row">
            <label htmlFor="energy-year">Year of temperature data</label>
            <div className="sim-param-input-wrap">
              <input
                type="number"
                id="energy-year"
                className="sim-param-input"
                value={energyYear}
                min={1950}
                max={2030}
                step={1}
                onChange={(e) => {
                  const v = parseInt(e.target.value, 10);
                  if (!isNaN(v)) setEnergyYear(v);
                }}
              />
            </div>
          </div>
          <div className="sim-actions">
            <button
              className="btn-primary btn-run"
              disabled={energyStatus?.kind === "running"}
              onClick={() => {
                if (!selectedBuilding) return;
                setEnergyStatus({ kind: "running", message: "Fetching temperature data…" });
                onAction("generate_temperature", {
                  lat: selectedBuilding.lat,
                  lon: selectedBuilding.lon,
                  year: energyYear,
                  bygningsnummer: selectedBuilding.bygningsnummer,
                });
              }}
            >
              ▶ Generate temperature data
            </button>
          </div>
          {energyStatus ? (
            <div className={`sim-status ${energyStatus.kind}`}>{energyStatus.message}</div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
