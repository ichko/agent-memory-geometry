/* Dataset validation and view calculations for precomputed, real-data atlases. */
export function validateAtlas(data) {
  const fail = message => { throw new Error(message); };
  if (!data || data.schemaVersion !== 2) fail("Open an atlas JSON built with tools/build_atlas.py (schemaVersion 2).");
  for (const key of ["points", "sessions", "histories", "systems"]) if (!Array.isArray(data[key])) fail(`Missing ${key} array.`);
  if (!data.points.length || !data.sessions.length || !data.histories.length) fail("The atlas must contain real points, conversations, and histories.");
  if (data.points.length > 1500000 || data.sessions.length > 100000 || data.histories.length > 10000) fail("This atlas is too large for this viewer.");
  if (typeof data.title !== "string" || !data.embedding || typeof data.embedding.model !== "string") fail("The atlas needs a title and embedding provenance.");
  if (!["analysis", "native"].includes(data.embedding.kind)) fail("This explorer expects real model embeddings, not synthetic vectors.");
  if (data.tsneDefault && !data.tsneLayouts?.[data.tsneDefault]) fail("The default t-SNE layout is not available in this atlas.");
  const pointId = value => Number.isInteger(value) && value >= 0 && value < data.points.length;
  for (const point of data.points) {
    if (typeof point.text !== "string") fail("Point text must be a string.");
    for (const projection of ["umap", "pca"]) if (!Array.isArray(point[projection]) || point[projection].length !== 2 || !point[projection].every(Number.isFinite)) fail(`Every point needs finite ${projection} coordinates.`);
    if (data.projections?.tsne && (!Array.isArray(point.tsne) || point.tsne.length !== 2 || !point.tsne.every(Number.isFinite))) fail("Every point needs finite tsne coordinates.");
    for (const layout of Object.values(data.tsneLayouts || {})) {
      if (typeof layout.pointKey !== "string" || !/^tsne(?:_[a-z][a-z0-9_]{0,23})?$/.test(layout.pointKey) || !Array.isArray(point[layout.pointKey]) || point[layout.pointKey].length !== 2 || !point[layout.pointKey].every(Number.isFinite)) fail("Every t-SNE comparison layout needs two finite coordinates per point.");
    }
    for (const method of Object.keys(data.projections3d || {})) {
      const xyz = point[`${method}3d`];
      if (!["umap", "pca"].includes(method) || !Array.isArray(xyz) || xyz.length !== 3 || !xyz.every(Number.isFinite)) fail("Every enabled 3D projection needs three finite coordinates per point.");
    }
  }
  for (const conversation of data.sessions) {
    if (typeof conversation.id !== "string" || typeof conversation.label !== "string" || !Array.isArray(conversation.turns) || !Array.isArray(conversation.histories)) fail("Invalid conversation record.");
    for (const turn of conversation.turns) if (!pointId(turn.p) || typeof turn.role !== "string") fail("Invalid conversation turn.");
    for (const id of conversation.histories) if (!Number.isInteger(id) || !data.histories[id]) fail("Unknown conversation history.");
  }
  const systemIds = new Set();
  for (const system of data.systems) {
    if (typeof system.id !== "string" || typeof system.label !== "string" || systemIds.has(system.id)) fail("Invalid or duplicate memory system.");
    systemIds.add(system.id);
  }
  for (const history of data.histories) {
    if (typeof history.question !== "string" || !pointId(history.query) || !Array.isArray(history.sessions) || !history.memories || typeof history.memories !== "object") fail("Invalid history record.");
    for (const id of history.sessions) if (!Number.isInteger(id) || !data.sessions[id]) fail("Unknown history conversation.");
    for (const [system, ids] of Object.entries(history.memories)) if (!systemIds.has(system) || !Array.isArray(ids) || !ids.every(pointId)) fail("Unknown memory point or system.");
  }
  return data;
}

export function projectionCoordinates(data, method, dimensions = 2, variant = null) {
  if (!["umap", "pca", "tsne"].includes(method)) throw new Error("Unknown projection.");
  if (method === "tsne" && !data.projections?.tsne) throw new Error("No t-SNE coordinates are available in this atlas.");
  if (method === "tsne" && variant && !data.tsneLayouts?.[variant]) throw new Error("Unknown t-SNE comparison layout.");
  if (dimensions === 3) {
    if (!data.projections3d?.[method]) throw new Error("No real 3D coordinates are available for this projection.");
    const key = `${method}3d`, minimum = [Infinity, Infinity, Infinity], maximum = [-Infinity, -Infinity, -Infinity];
    for (const point of data.points) for (let axis = 0; axis < 3; axis++) {
      minimum[axis] = Math.min(minimum[axis], point[key][axis]); maximum[axis] = Math.max(maximum[axis], point[key][axis]);
    }
    const span = Math.max(...maximum.map((value, axis) => value - minimum[axis]), 1e-9);
    return data.points.map(point => point[key].map((value, axis) => (value - (minimum[axis] + maximum[axis]) / 2) / span * (axis === 1 ? -1 : 1)));
  }
  const key = method === "tsne" && variant ? data.tsneLayouts[variant].pointKey : method;
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const point of data.points) {
    const [x, y] = point[key];
    minX = Math.min(minX, x); maxX = Math.max(maxX, x);
    minY = Math.min(minY, y); maxY = Math.max(maxY, y);
  }
  const span = Math.max(maxX - minX, maxY - minY, 1e-9);
  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
  return data.points.map(point => [(point[key][0] - cx) / span, -(point[key][1] - cy) / span]);
}

export function visibleSessions(data, historyIndex = null, search = "") {
  const ids = historyIndex === null ? data.sessions.map((_, i) => i) : data.histories[historyIndex].sessions;
  const needle = search.trim().toLowerCase();
  return needle ? ids.filter(id => data.sessions[id].turns.some(turn => data.points[turn.p].text.toLowerCase().includes(needle))) : ids;
}

export function memoryPoints(data, systemId, historyIndex = null) {
  if (!systemId) return [];
  const histories = historyIndex === null ? data.histories : [data.histories[historyIndex]];
  return [...new Set(histories.flatMap(history => history.memories[systemId] || []))];
}

export function stepHistory(current, total, direction) {
  if (total < 1) return null;
  if (current === null) return direction < 0 ? total - 1 : 0;
  return (current + direction + total) % total;
}

export function randomHistory(current, total, random = Math.random) {
  if (total < 1) return null;
  if (total === 1) return 0;
  const choice = Math.floor(random() * (total - (current === null ? 0 : 1)));
  return current !== null && choice >= current ? choice + 1 : choice;
}

export function gridLines(size, zoom, pan, baseScale) {
  return gridMarks(size, zoom, pan, baseScale).map(mark => mark.position);
}

export function gridMarks(size, zoom, pan, baseScale) {
  const scale = zoom * baseScale;
  const step = 2 ** Math.floor(Math.log2(80 / scale));
  const fade = Math.max(0, Math.min(1, Math.log2(step * scale / 40)));
  const start = (-size / 2 - pan) / scale, end = (size / 2 - pan) / scale;
  const lines = [];
  for (let index = Math.ceil(start / step); index <= Math.floor(end / step); index++) {
    lines.push({ position: size / 2 + pan + index * step * scale, opacity: index % 2 === 0 ? 1 : fade });
  }
  return lines;
}
