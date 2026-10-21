import { orbitRotation } from "./spatial.mjs";

export function previewShouldAnimate({ visible, hidden, spinning, dragging, dimensions, hovering = false }) {
  return visible && !hidden && spinning && !dragging && !hovering && dimensions === 3;
}

export function previewAutoZoom(elapsedMs) {
  return 1 + .06 * Math.sin(elapsedMs * 2 * Math.PI / 60000);
}

export function previewAutoOrbit(elapsedMs, deltaMs) {
  const phase = 2 * Math.PI / 48000;
  return { yaw: deltaMs * .00012, pitch: .32 * (Math.sin((elapsedMs + deltaMs) * phase) - Math.sin(elapsedMs * phase)) };
}

export function previewBubblePosition([x, y], width, height, bounds, blocked = []) {
  const gap = 24;
  const clamp = (v, low, high) => Math.max(low, Math.min(high, v));
  const centeredX = clamp(x - width / 2, bounds.left, bounds.right - width);
  const centeredY = clamp(y - height / 2, bounds.top, bounds.bottom - height);
  const candidates = [
    { x: x + gap, y: centeredY }, { x: x - gap - width, y: centeredY },
    { x: centeredX, y: y + gap }, { x: centeredX, y: y - gap - height },
  ];
  const fits = p => p.x >= bounds.left && p.x + width <= bounds.right && p.y >= bounds.top && p.y + height <= bounds.bottom;
  const overlap = p => blocked.reduce((area, r) => area + Math.max(0, Math.min(p.x + width, r.right) - Math.max(p.x, r.left)) * Math.max(0, Math.min(p.y + height, r.bottom) - Math.max(p.y, r.top)), 0);
  const available = candidates.filter(fits);
  return available.sort((a, b) => overlap(a) - overlap(b))[0] ?? candidates[2];
}

export function previewMemoryHighlight(memories, elapsedMs) {
  if (!memories.length) return null;
  const count = Math.min(6, memories.length), cycle = 14000;
  const slot = Math.floor(elapsedMs / cycle) % count;
  const id = memories[Math.round(slot * (memories.length - 1) / Math.max(1, count - 1))];
  const time = elapsedMs % cycle;
  return { id, opacity: 1, radius: 7 + 1.5 * Math.sin(time * 2 * Math.PI / 6000) };
}

export function previewFrontMemory(details, positions, width, height, previousId = null) {
  const visible = details.filter(memory => {
    const p = positions[memory.point];
    return memory.text && memory.matches.length && p && p[0] >= 30 && p[0] <= width - 25 && p[1] >= 22 && p[1] <= height - 22;
  }).sort((a, b) => positions[b.point][2] - positions[a.point][2] || a.point - b.point);
  // Retain the foreground preference even when cycling away from the last point.
  const front = visible.filter(m => positions[m.point][2] >= (positions[visible[0]?.point]?.[2] ?? 0) - .15);
  return front.find(m => m.point !== previousId) ?? front[0] ?? null;
}

export function preparePreview(data, key) {
  const layout = data.layouts[key];
  if (!layout || ![2, 3].includes(layout.dimensions)) throw new Error("Missing measured preview layout");
  const coordinates = layout.coordinates;
  if (!coordinates.length || coordinates.length !== data.pointIds.length || coordinates.some(p => p.length !== layout.dimensions || p.some(v => !Number.isFinite(v)))) throw new Error("Invalid measured preview coordinates");
  for (const id of [...data.turns, ...data.memories, ...data.paths.flat()]) if (!Number.isInteger(id) || id < 0 || id >= coordinates.length) throw new Error("Invalid preview point ID");
  for (const id of data.backgroundPaths ?? []) if (!Number.isInteger(id) || !data.paths[id]) throw new Error("Invalid preview background path");
  const memoryIds = new Set(data.memories), detailIds = new Set();
  for (const detail of data.memoryDetails ?? []) {
    if (!memoryIds.has(detail.point) || detailIds.has(detail.point) || typeof detail.text !== "string" || !Array.isArray(detail.matches)) throw new Error("Invalid preview memory detail");
    detailIds.add(detail.point);
    for (const match of detail.matches) if (!Number.isInteger(match.path) || !data.paths[match.path]?.includes(match.point) || !Number.isFinite(match.score) || Math.abs(match.score) > 1) throw new Error("Invalid original-space memory match");
  }
  const center = Array.from({ length: layout.dimensions }, (_, axis) => (Math.min(...coordinates.map(p => p[axis])) + Math.max(...coordinates.map(p => p[axis]))) / 2);
  const centered = coordinates.map(p => p.map((v, axis) => v - center[axis]));
  const radius = Math.max(...centered.map(p => Math.hypot(...p)), 1e-9);
  return { ...layout, coordinates: centered.map(p => p.map(v => v / radius)) };
}

export function previewProjection(points, dimensions, yaw, pitch, width, height, zoom = 1, pan = [0, 0]) {
  const rotation = dimensions === 3 ? orbitRotation(yaw, pitch) : null;
  const scale = Math.min(width, height) * .435 * zoom;
  return points.map(([x, y, z = 0]) => {
    const rx = rotation ? rotation[0] * x + rotation[3] * y + rotation[6] * z : x;
    const ry = rotation ? rotation[1] * x + rotation[4] * y + rotation[7] * z : y;
    const depth = rotation ? rotation[2] * x + rotation[5] * y + rotation[8] * z : 0;
    return [width / 2 + pan[0] + rx * scale, height / 2 + pan[1] - ry * scale, depth];
  });
}

export function previewDrag(dimensions, origin, dx, dy) {
  return dimensions === 3
    ? { ...origin, yaw: origin.yaw + dx * .009, pitch: Math.max(-1.45, Math.min(1.45, origin.pitch + dy * .009)) }
    : { ...origin, pan: [origin.pan[0] + dx, origin.pan[1] + dy] };
}
