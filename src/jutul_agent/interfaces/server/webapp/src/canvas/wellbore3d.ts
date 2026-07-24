// A custom MapLibre WebGL layer that draws a well's bore as stacked tapered
// cylinder segments above its surface point, colour-ramped by depth, with a
// red wellhead cap and a rise-up animation. AGS (a single energy well) draws
// one column; BTES (a well park) draws a hexagonal array. Ported from
// geothermal-viz's web/js/wellbore-3d.js, with its module-level singleton
// replaced by one controller per map instance (created in MapPanel.tsx's map
// mount effect) — the same imperative-controller-outside-React pattern that
// file already uses for the map and popup themselves.

import maplibregl from "maplibre-gl";

const LAYER_ID = "wellbore-3d";

const SEGMENT_GAP_FRAC = 0.2; // fraction of segment height used as gap
const VERTICAL_OFFSET = 40; // metres above surface for well base
const WELL_RADIUS = 8; // radius of single-well column (m)
const PARK_WELL_RADIUS = 3.5; // radius per well in a wellpark (m)
const CAP_HEIGHT_FRAC = 0.025; // wellhead cap height as fraction of depth
const N_SIDES = 16; // smooth cylindrical cross-section
const MIN_SEGMENTS = 2;
const MAX_SEGMENTS = 100;
const MAX_RENDERED_WELLS = 80;
const ANIM_DURATION = 1200;
const ANIM_DELAY = 500;
const FLY_ZOOM = 15.5;
const FLY_PITCH = 60;

type Color = [number, number, number];

const CAP_COLOR: Color = [0.85, 0.22, 0.18]; // red wellhead

/** Depth-based colour ramp: warm deep -> cool shallow (t: 0 = deep, 1 = shallow). */
function depthColor(t: number): Color {
  if (t < 0.5) {
    const s = t * 2;
    return [0.72 + (0.2 - 0.72) * s, 0.38 + (0.55 - 0.38) * s, 0.18 + (0.65 - 0.18) * s];
  }
  const s = (t - 0.5) * 2;
  return [0.2 + (0.48 - 0.2) * s, 0.55 + (0.78 - 0.55) * s, 0.65 + (0.92 - 0.65) * s];
}

const VS_SRC = `
  attribute vec3 a_pos;
  attribute vec4 a_color;
  uniform mat4 u_matrix;
  varying vec4 v_color;
  void main() {
    gl_Position = u_matrix * vec4(a_pos, 1.0);
    v_color = a_color;
  }
`;

const FS_SRC = `
  precision mediump float;
  varying vec4 v_color;
  void main() {
    gl_FragColor = v_color;
  }
`;

function compileShader(gl: WebGLRenderingContext, type: number, src: string): WebGLShader | null {
  const s = gl.createShader(type);
  if (!s) return null;
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    console.error("Shader compile error:", gl.getShaderInfoLog(s));
    gl.deleteShader(s);
    return null;
  }
  return s;
}

function linkProgram(gl: WebGLRenderingContext, vsSrc: string, fsSrc: string): WebGLProgram | null {
  const vs = compileShader(gl, gl.VERTEX_SHADER, vsSrc);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, fsSrc);
  if (!vs || !fs) return null;
  const p = gl.createProgram();
  if (!p) return null;
  gl.attachShader(p, vs);
  gl.attachShader(p, fs);
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    console.error("Program link error:", gl.getProgramInfoLog(p));
    gl.deleteProgram(p);
    return null;
  }
  return p;
}

interface HexPos {
  dx: number;
  dy: number;
}

/** Hexagonal layout positions (metre offsets) for a wellpark. */
function hexLayout(count: number, spacing: number): HexPos[] {
  if (count <= 0) return [];
  const out: HexPos[] = [{ dx: 0, dy: 0 }];
  let ring = 1;
  while (out.length < count) {
    for (let side = 0; side < 6; side++) {
      for (let j = 0; j < ring; j++) {
        if (out.length >= count) return out;
        const a0 = (side * Math.PI) / 3;
        const a1 = (((side + 1) % 6) * Math.PI) / 3;
        const t = ring > 1 ? j / ring : 0;
        out.push({
          dx: ring * spacing * Math.cos(a0) + t * ring * spacing * (Math.cos(a1) - Math.cos(a0)),
          dy: ring * spacing * Math.sin(a0) + t * ring * spacing * (Math.sin(a1) - Math.sin(a0)),
        });
      }
    }
    ring++;
  }
  return out;
}

interface Point2D {
  x: number;
  y: number;
}

function polyCircle(n: number): Point2D[] {
  const pts: Point2D[] = [];
  for (let i = 0; i < n; i++) {
    const a = (2 * Math.PI * i) / n;
    pts.push({ x: Math.cos(a), y: Math.sin(a) });
  }
  return pts;
}

const CIRCLE = polyCircle(N_SIDES);

/** Push a single vertex into the buffer: 3 pos + 4 colour/alpha. */
function v(arr: number[], x: number, y: number, z: number, r: number, g: number, b: number, a: number): void {
  arr.push(x, y, z, r, g, b, a);
}

/** A tapered tube (wider at bottom, narrower at top, no caps) — the taper
 *  makes individual stacked segments clearly distinguishable. */
function addTube(
  arr: number[],
  cx: number,
  cy: number,
  z0: number,
  z1: number,
  radiusBot: number,
  col: Color,
  alpha: number,
): void {
  const pts = CIRCLE;
  const n = pts.length;
  const cr = col[0] * 0.7;
  const cg = col[1] * 0.7;
  const cb = col[2] * 0.7;
  const radiusTop = radiusBot * 0.85;

  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    const ax0 = cx + pts[i].x * radiusBot;
    const ay0 = cy + pts[i].y * radiusBot;
    const bx0 = cx + pts[j].x * radiusBot;
    const by0 = cy + pts[j].y * radiusBot;
    const ax1 = cx + pts[i].x * radiusTop;
    const ay1 = cy + pts[i].y * radiusTop;
    const bx1 = cx + pts[j].x * radiusTop;
    const by1 = cy + pts[j].y * radiusTop;

    v(arr, ax0, ay0, z0, cr, cg, cb, alpha);
    v(arr, bx0, by0, z0, cr, cg, cb, alpha);
    v(arr, bx1, by1, z1, cr, cg, cb, alpha);

    v(arr, ax0, ay0, z0, cr, cg, cb, alpha);
    v(arr, bx1, by1, z1, cr, cg, cb, alpha);
    v(arr, ax1, ay1, z1, cr, cg, cb, alpha);
  }
}

/** A capped cylinder (top cap + side walls) — only used for the wellhead cap,
 *  where the top must actually be visible. */
function addCappedCylinder(
  arr: number[],
  cx: number,
  cy: number,
  z0: number,
  z1: number,
  radius: number,
  col: Color,
  alpha: number,
): void {
  const pts = CIRCLE;
  const n = pts.length;
  const [cr, cg, cb] = col;

  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    v(arr, cx, cy, z1, cr, cg, cb, alpha);
    v(arr, cx + pts[i].x * radius, cy + pts[i].y * radius, z1, cr, cg, cb, alpha);
    v(arr, cx + pts[j].x * radius, cy + pts[j].y * radius, z1, cr, cg, cb, alpha);
  }

  const sr = cr * 0.7;
  const sg = cg * 0.7;
  const sb = cb * 0.7;
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    const ax = cx + pts[i].x * radius;
    const ay = cy + pts[i].y * radius;
    const bx = cx + pts[j].x * radius;
    const by = cy + pts[j].y * radius;

    v(arr, ax, ay, z0, sr, sg, sb, alpha);
    v(arr, bx, by, z0, sr, sg, sb, alpha);
    v(arr, bx, by, z1, sr, sg, sb, alpha);

    v(arr, ax, ay, z0, sr, sg, sb, alpha);
    v(arr, bx, by, z1, sr, sg, sb, alpha);
    v(arr, ax, ay, z1, sr, sg, sb, alpha);
  }
}

interface Mercator {
  x: number;
  y: number;
}

/** Build all vertex data (pos + colour) for the well visualisation, fully
 *  opaque: segment cylinders plus a wellhead cap. */
function buildVertices(
  center: Mercator,
  scale: number,
  params: Record<string, number>,
  caseType: string | null,
  progress: number,
  groundElevation: number,
): Float32Array {
  const verts: number[] = [];
  const depth = params.well_depth || 200;
  const nSeg = Math.max(MIN_SEGMENTS, Math.min(Math.round(params.num_segments || 10), MAX_SEGMENTS));

  let positions: HexPos[];
  let baseRadius: number;
  if (caseType === "BTES") {
    const n = Math.min(params.num_wells_btes || 48, MAX_RENDERED_WELLS);
    const sp = params.well_spacing || 5;
    baseRadius = PARK_WELL_RADIUS * scale;
    const displaySpacing = Math.max(PARK_WELL_RADIUS * 3, sp * (n > 20 ? 4 : 6));
    positions = hexLayout(n, displaySpacing);
  } else {
    positions = [{ dx: 0, dy: 0 }];
    baseRadius = WELL_RADIUS * scale;
  }

  const segTotal = depth / nSeg;
  const gapHeight = segTotal * SEGMENT_GAP_FRAC;
  const segHeight = segTotal - gapHeight;

  // Floats VERTICAL_OFFSET metres above the *local* ground, not sea level —
  // with terrain enabled (see MapPanel.tsx's setTerrain), a well anywhere
  // above ~40m elevation would otherwise sit behind (be depth-occluded by)
  // the terrain mesh, rendering correctly but never actually visible.
  const base = groundElevation + VERTICAL_OFFSET;

  for (const pos of positions) {
    const cx = center.x + pos.dx * scale;
    const cy = center.y - pos.dy * scale;
    let curZ = base;

    for (let i = 0; i < nSeg; i++) {
      const t = nSeg > 1 ? i / (nSeg - 1) : 0.5;
      const col = depthColor(t);
      const z0 = curZ * scale * progress;
      const z1 = (curZ + segHeight) * scale * progress;
      if (z1 > z0 + 1e-12) addTube(verts, cx, cy, z0, z1, baseRadius, col, 1.0);
      curZ += segTotal;
    }

    const capH = Math.max(depth * CAP_HEIGHT_FRAC, 3);
    const capZ0 = (base + depth) * scale * progress;
    const capZ1 = (base + depth + capH) * scale * progress;
    const capR = baseRadius * 1.25;
    if (capZ1 > capZ0 + 1e-12) addCappedCylinder(verts, cx, cy, capZ0, capZ1, capR, CAP_COLOR, 1.0);
  }

  return new Float32Array(verts);
}

/** One borehole's position (metres relative to the field origin) and depth. */
export interface BoreholePos {
  x: number;
  y: number;
  H: number;
  D?: number;           // buried depth from surface to top of active section [m] (default 0.5)
  tilt?: number;        // radians from vertical (0 = vertical)
  orientation?: number; // azimuthal direction of tilt [radians], 0 = +x (east)
}

/** Build all vertex data for a field of boreholes at explicit positions.
 *  displayScale inflates every metre offset uniformly so the field is visible
 *  at FLY_ZOOM — the same concept as the BTES displaySpacing inflation. */
function buildFieldVertices(
  center: Mercator,
  scale: number,
  boreholes: BoreholePos[],
  displayScale: number,
  progress: number,
  groundElevation: number,
): Float32Array {
  const verts: number[] = [];
  const baseRadius = PARK_WELL_RADIUS * scale;
  const base = groundElevation + VERTICAL_OFFSET;

  for (const bh of boreholes) {
    // Top position using the same displayScale as before.
    const cx_top = center.x + bh.x * displayScale * scale;
    const cy_top = center.y - bh.y * displayScale * scale; // Mercator y increases southward

    // Tilt: horizontal displacement from top to bottom in real metres (no displayScale),
    // orientation=0 is +x (east), matching the borehole coordinate convention.
    const tilt = bh.tilt ?? 0;
    const orientation = bh.orientation ?? 0;
    const sinT = Math.sin(tilt);
    const hx = bh.H * sinT * Math.cos(orientation) * scale; // actual metres, not inflated
    const hy = bh.H * sinT * Math.sin(orientation) * scale;

    const H = bh.H;
    const D = bh.D ?? 0.5;

    // Subtract D from the starting z so a larger D shifts the whole column downward —
    // the active section runs from z=base-D (deep end) to z=base-D+H (top of active section).
    const nSeg = Math.max(MIN_SEGMENTS, Math.min(Math.round(H / 20), MAX_SEGMENTS));
    const segTotal = H / nSeg;
    const segHeight = segTotal * (1 - SEGMENT_GAP_FRAC);

    let curZ = base - D;
    for (let i = 0; i < nSeg; i++) {
      const t_mid = (i + 0.5) / nSeg;
      const cx = cx_top + (1 - t_mid) * hx;
      const cy = cy_top - (1 - t_mid) * hy;
      const t = nSeg > 1 ? i / (nSeg - 1) : 0.5;
      const col = depthColor(t);
      const z0 = curZ * scale * progress;
      const z1 = (curZ + segHeight) * scale * progress;
      if (z1 > z0 + 1e-12) addTube(verts, cx, cy, z0, z1, baseRadius, col, 1.0);
      curZ += segTotal;
    }

    // Cap at the top of the active section (base-D+H); goes lower as D increases.
    const capH = Math.max(H * CAP_HEIGHT_FRAC, 3);
    const capZ0 = (base - D + H) * scale * progress;
    const capZ1 = (base - D + H + capH) * scale * progress;
    if (capZ1 > capZ0 + 1e-12) addCappedCylinder(verts, cx_top, cy_top, capZ0, capZ1, baseRadius * 1.25, CAP_COLOR, 1.0);
  }

  return new Float32Array(verts);
}

export interface Wellbore3D {
  /** Show (or replace) the 3D wellbore at `lngLat`, animating it rising up. */
  show(lngLat: { lng: number; lat: number }, params: Record<string, number>, caseType: string | null): void;
  /** Show (or replace) a borehole field at `lngLat` using explicit positions. */
  showField(lngLat: { lng: number; lat: number }, boreholes: BoreholePos[]): void;
  /** Live-update field borehole positions without re-animating (for real-time param changes). */
  updateField(boreholes: BoreholePos[]): void;
  /** Live-update displayed parameters (e.g. an edited depth) without re-animating. */
  update(params: Record<string, number>): void;
  /** Remove the layer; safe to call when nothing is shown. */
  remove(): void;
}

/** One controller per map instance. The GL program/buffer are created lazily
 *  by MapLibre's own `onAdd` the first time the layer is added; `remove()`
 *  only removes the layer (mirroring MapPanel.tsx's own map/popup teardown,
 *  which leans on `map.remove()` to release the GL context on unmount rather
 *  than freeing buffers by hand). */
export function createWellbore3D(map: maplibregl.Map): Wellbore3D {
  const state = {
    active: false,
    lngLat: null as { lng: number; lat: number } | null,
    params: null as Record<string, number> | null,
    caseType: null as string | null,
    boreholes: null as BoreholePos[] | null,
    fieldDisplayScale: 1, // uniform scale applied to positions + radius in showField
    progress: 0,
    animStart: 0,
    needsBuild: false,
    program: null as WebGLProgram | null,
    buffer: null as WebGLBuffer | null,
    vertCount: 0,
    aPos: -1,
    aColor: -1,
    uMatrix: null as WebGLUniformLocation | null,
  };

  function rebuild(gl: WebGLRenderingContext): void {
    if (!state.lngLat || !state.buffer) return;
    const mc = maplibregl.MercatorCoordinate.fromLngLat(state.lngLat, 0);
    const scale = mc.meterInMercatorCoordinateUnits();
    const groundElevation = map.queryTerrainElevation(state.lngLat) ?? 0;
    const vertices = state.boreholes
      ? buildFieldVertices(mc, scale, state.boreholes, state.fieldDisplayScale, state.progress, groundElevation)
      : buildVertices(mc, scale, state.params!, state.caseType, state.progress, groundElevation);
    gl.bindBuffer(gl.ARRAY_BUFFER, state.buffer);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.DYNAMIC_DRAW);
    state.vertCount = vertices.length / 7;
  }

  const customLayer: maplibregl.CustomLayerInterface = {
    id: LAYER_ID,
    type: "custom",
    renderingMode: "3d",

    onAdd(_map: maplibregl.Map, gl: WebGLRenderingContext) {
      state.program = linkProgram(gl, VS_SRC, FS_SRC);
      if (!state.program) return;
      state.aPos = gl.getAttribLocation(state.program, "a_pos");
      state.aColor = gl.getAttribLocation(state.program, "a_color");
      state.uMatrix = gl.getUniformLocation(state.program, "u_matrix");
      state.buffer = gl.createBuffer();
      state.needsBuild = true;
    },

    // maplibre-gl v5's render signature is `(gl, options)`, not the v1-style
    // `(gl, matrix)`. The matrix that accepts plain MercatorCoordinate-space
    // (normalized 0..1) vertices — the convention this file's geometry uses
    // throughout — is `options.defaultProjectionData.mainMatrix`, confirmed
    // against MapLibre's own "Add a custom style layer" example.
    // `options.modelViewProjectionMatrix` is a *different* matrix with a
    // different expected input scale; using it instead silently projects
    // everything far outside the viewport (no error, nothing drawn).
    render(gl: WebGLRenderingContext, options: maplibregl.CustomRenderMethodInput) {
      if (!state.active || !state.program) return;

      let animating = false;
      if (state.progress < 1) {
        const t = Math.max(0, Math.min((performance.now() - state.animStart) / ANIM_DURATION, 1));
        state.progress = 1 - Math.pow(1 - t, 3); // ease-out cubic
        state.needsBuild = true;
        animating = true;
      }

      if (state.needsBuild) {
        rebuild(gl);
        state.needsBuild = false;
      }
      if (state.vertCount === 0) return;

      gl.useProgram(state.program);
      gl.uniformMatrix4fv(state.uMatrix, false, options.defaultProjectionData.mainMatrix as Float32Array);
      gl.enable(gl.DEPTH_TEST);
      gl.depthMask(true);
      gl.disable(gl.BLEND);

      const stride = 7 * 4;
      gl.bindBuffer(gl.ARRAY_BUFFER, state.buffer);
      gl.enableVertexAttribArray(state.aPos);
      gl.vertexAttribPointer(state.aPos, 3, gl.FLOAT, false, stride, 0);
      gl.enableVertexAttribArray(state.aColor);
      gl.vertexAttribPointer(state.aColor, 4, gl.FLOAT, false, stride, 12);
      gl.drawArrays(gl.TRIANGLES, 0, state.vertCount);

      if (animating) map.triggerRepaint();
    },

    onRemove(_map: maplibregl.Map, gl: WebGLRenderingContext) {
      if (state.buffer) {
        gl.deleteBuffer(state.buffer);
        state.buffer = null;
      }
      if (state.program) {
        gl.deleteProgram(state.program);
        state.program = null;
      }
      state.vertCount = 0;
    },
  };

  function remove(): void {
    if (map.getLayer(LAYER_ID)) {
      try {
        map.removeLayer(LAYER_ID);
      } catch {
        // already removed
      }
    }
    state.active = false;
    state.vertCount = 0;
    state.params = null;
    state.boreholes = null;
    state.lngLat = null;
    map.triggerRepaint();
  }

  function show(lngLat: { lng: number; lat: number }, params: Record<string, number>, caseType: string | null): void {
    remove();

    state.lngLat = lngLat;
    state.params = { ...params };
    state.caseType = caseType;
    state.active = true;
    state.progress = 0;
    state.animStart = performance.now() + ANIM_DELAY;

    if (!map.getLayer(LAYER_ID)) map.addLayer(customLayer);

    map.flyTo({ center: lngLat, zoom: FLY_ZOOM, pitch: FLY_PITCH, duration: 1200, essential: true });
    map.triggerRepaint();
  }

  function showField(lngLat: { lng: number; lat: number }, boreholes: BoreholePos[]): void {
    remove();

    state.lngLat = lngLat;
    state.boreholes = boreholes;
    state.active = true;
    state.progress = 0;
    state.animStart = performance.now() + ANIM_DELAY;

    if (!map.getLayer(LAYER_ID)) map.addLayer(customLayer);

    // Inflate the field uniformly (positions + radius) using the same displaySpacing
    // logic as the BTES hex layout, so the field occupies similar screen real estate
    // at FLY_ZOOM. Fly to the actual geographic centroid of the field (not scaled).
    let flyLng = lngLat.lng;
    let flyLat = lngLat.lat;
    state.fieldDisplayScale = 1;
    if (boreholes.length > 1) {
      const xs = boreholes.map((b) => b.x);
      const ys = boreholes.map((b) => b.y);
      const xMid = (Math.min(...xs) + Math.max(...xs)) / 2;
      const yMid = (Math.min(...ys) + Math.max(...ys)) / 2;
      const cosLat = Math.cos(lngLat.lat * (Math.PI / 180));
      flyLat = lngLat.lat + yMid / 111320;
      flyLng = lngLat.lng + xMid / (111320 * cosLat);

      // Minimum spacing between any two boreholes (O(N²), N is small in practice).
      let minSpacing = Infinity;
      for (let i = 0; i < boreholes.length; i++) {
        for (let j = i + 1; j < boreholes.length; j++) {
          const dx = boreholes[i].x - boreholes[j].x;
          const dy = boreholes[i].y - boreholes[j].y;
          const d = Math.sqrt(dx * dx + dy * dy);
          if (d < minSpacing) minSpacing = d;
        }
      }
      if (minSpacing > 0 && isFinite(minSpacing)) {
        // Match the BTES display spacing: inflate so the nearest-neighbour gap
        // looks like a BTES park of the same size at the standard zoom.
        const n = boreholes.length;
        const displaySpacing = Math.max(PARK_WELL_RADIUS * 3, minSpacing * (n > 20 ? 4 : 6));
        state.fieldDisplayScale = displaySpacing / minSpacing;
      }
    }

    map.flyTo({ center: [flyLng, flyLat], zoom: FLY_ZOOM, pitch: FLY_PITCH, duration: 1200, essential: true });
    map.triggerRepaint();
  }

  function updateField(boreholes: BoreholePos[]): void {
    if (!state.active) return;
    state.boreholes = boreholes;
    if (boreholes.length > 1) {
      let minSpacing = Infinity;
      for (let i = 0; i < boreholes.length; i++) {
        for (let j = i + 1; j < boreholes.length; j++) {
          const dx = boreholes[i].x - boreholes[j].x;
          const dy = boreholes[i].y - boreholes[j].y;
          const d = Math.sqrt(dx * dx + dy * dy);
          if (d < minSpacing) minSpacing = d;
        }
      }
      if (minSpacing > 0 && isFinite(minSpacing)) {
        const n = boreholes.length;
        const displaySpacing = Math.max(PARK_WELL_RADIUS * 3, minSpacing * (n > 20 ? 4 : 6));
        state.fieldDisplayScale = displaySpacing / minSpacing;
      }
    }
    state.progress = 1;
    state.needsBuild = true;
    map.triggerRepaint();
  }

  function update(params: Record<string, number>): void {
    if (!state.active || !state.params) return;
    Object.assign(state.params, params);
    state.progress = 1;
    state.needsBuild = true;
    map.triggerRepaint();
  }

  return { show, showField, updateField, update, remove };
}
