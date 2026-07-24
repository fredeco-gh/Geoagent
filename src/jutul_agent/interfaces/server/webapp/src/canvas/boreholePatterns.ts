import type { BoreholePos } from "./wellbore3d";

export function rectangleField(N1: number, N2: number, B: number, H: number, D: number): BoreholePos[] {
  const cx = ((N1 - 1) * B) / 2;
  const cy = ((N2 - 1) * B) / 2;
  const out: BoreholePos[] = [];
  for (let i = 0; i < N1; i++)
    for (let j = 0; j < N2; j++)
      out.push({ x: i * B - cx, y: j * B - cy, H, D });
  return out;
}

export function sunflowerField(N: number, B: number, H: number, D: number): BoreholePos[] {
  if (N <= 0) return [];
  const phi = (1 + Math.sqrt(5)) / 2;
  const dTheta = (2 * Math.PI) / (phi * phi);
  // Scale radius so nearest-neighbour spacing matches B (mirrors pygfunction_sim.py)
  const p1x = Math.sqrt(1 / N) * Math.cos(dTheta);
  const p1y = Math.sqrt(1 / N) * Math.sin(dTheta);
  const curSpacing = Math.sqrt(p1x * p1x + p1y * p1y) || 1;
  const radius = (1.3513 * B) / curSpacing;
  return Array.from({ length: N }, (_, i) => {
    const r = Math.sqrt(i / N) * radius;
    const theta = i * dTheta;
    return { x: r * Math.cos(theta), y: r * Math.sin(theta), H, D };
  });
}

export function circularField(N: number, B: number, H: number, D: number): BoreholePos[] {
  if (N <= 0) return [];
  const pts: [number, number][] = [[0, 0]];
  let ring = 1;
  while (pts.length < N) {
    const r = ring * B;
    const nRing = Math.min(Math.max(1, Math.round((2 * Math.PI * r) / B)), N - pts.length);
    for (let k = 0; k < nRing; k++) {
      const theta = (2 * Math.PI * k) / nRing;
      pts.push([r * Math.cos(theta), r * Math.sin(theta)]);
    }
    ring++;
  }
  return pts.slice(0, N).map(([x, y]) => ({ x, y, H, D }));
}

function pointInPolygon(x: number, y: number, poly: [number, number][]): boolean {
  let inside = false;
  let j = poly.length - 1;
  for (let i = 0; i < poly.length; i++) {
    const [xi, yi] = poly[i];
    const [xj, yj] = poly[j];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
    j = i;
  }
  return inside;
}

export function polygonalField(N: number, numSides: number, B: number, H: number, D: number): BoreholePos[] {
  if (N <= 0 || numSides < 3) return [];
  const R = Math.sqrt((2 * N * B * B) / (numSides * Math.sin((2 * Math.PI) / numSides)));
  const poly: [number, number][] = Array.from({ length: numSides }, (_, k) => {
    const theta = (2 * Math.PI * k) / numSides;
    return [R * Math.cos(theta), R * Math.sin(theta)];
  });
  const margin = 1.2;
  const nxy = 2 * Math.ceil((margin * R) / B) + 1;
  const start = -((nxy - 1) / 2) * B;
  const candidates: [number, number][] = [];
  for (let i = 0; i < nxy; i++) {
    for (let j = 0; j < nxy; j++) {
      const x = start + i * B;
      const y = start + j * B;
      if (pointInPolygon(x, y, poly)) candidates.push([x, y]);
    }
  }
  candidates.sort((a, b) => Math.hypot(a[0], a[1]) - Math.hypot(b[0], b[1]));
  return candidates.slice(0, N).map(([x, y]) => ({ x, y, H, D }));
}

export function computeBtesBoreholes(
  pattern: string,
  n: number,
  n1: number,
  n2: number,
  numSides: number,
  spacing: number,
  H: number,
  D: number,
): BoreholePos[] {
  switch (pattern) {
    case "rectangular": return rectangleField(n1, n2, spacing, H, D);
    case "circular":    return circularField(n, spacing, H, D);
    case "polygonal":   return polygonalField(n, numSides, spacing, H, D);
    default:            return sunflowerField(n, spacing, H, D);
  }
}
