import { validateAtlas, projectionCoordinates, visibleSessions, memoryPoints, gridMarks, stepHistory, randomHistory } from "./atlas.mjs";
import { wheelZoomFactor, zoomViewport, fitPoints } from "./navigation.mjs?v=3";
import { createPointRenderer, ATLAS_COLORS, PLOT_STYLE, markerStyle, markerHitRadius, pathOpacity, contextPathWidth } from "./points.mjs";
import { guidedExamples } from "./examples.mjs";
import { buildPathIndex, nearestPath } from "./paths.mjs";
import { createMemoryInspector } from "./inspector.mjs";
import { unpackAtlas } from "./data-loader.mjs";
import { createSpatialPaths, IDENTITY_ROTATION, orbitRotation, rotatedXY, coordinateMoments, depthCueForView } from "./spatial.mjs";

const $ = id => document.getElementById(id);
const canvas = $("atlas-canvas"), ctx = canvas.getContext("2d");
const rendererOptions = { backend: new URLSearchParams(location.search).get("renderer") === "canvas" ? "canvas" : "auto" };
const gridCanvas = $("grid-canvas"), gridCtx = gridCanvas.getContext("2d");
const pointCanvas = $("point-canvas"), pointRenderer = createPointRenderer(pointCanvas, rendererOptions);
const pathCanvas = $("path-canvas"), pathCtx = pathCanvas.getContext("2d");
const spatialCanvas = $("spatial-path-canvas"), spatialPaths = createSpatialPaths(spatialCanvas);
const memoryCanvas = $("memory-canvas"), memoryRenderer = createPointRenderer(memoryCanvas, rendererOptions);
const state = { data: null, method: "umap", xy: [], history: null, system: "", search: "", sessions: [], memories: [], zoom: 1, pan: [0, 0], selected: null, selectedPoint: null, hover: null, width: 0, height: 0, baseScale: 1, paths: [], pathKey: null, turnPoints: [], buckets: new Map(), pending: false, cloudOpacity: PLOT_STYLE.cloudOpacity, memoryOpacity: PLOT_STYLE.memoryOpacity, markerArea: PLOT_STYLE.area };
state.autoOpacity = true;
state.examples = []; state.example = null; state.beforeExample = null;
state.dimension = 2; state.orbit = [-.45, .3]; state.rotation = IDENTITY_ROTATION; state.pickRotation = null;
state.depthShading = true; state.moments = null;
state.pickCoordinates = []; state.pathPicking = null;
state.links = [];
state.tsneLayout = null;
const number = value => value.toLocaleString("en-US");
const el = (tag, text = "", className = "") => { const node = document.createElement(tag); node.textContent = text; if (className) node.className = className; return node; };
const option = (value, label) => { const node = el("option", label); node.value = value; return node; };
const announce = text => { $("announcement").textContent = text; };
const profiling = new URLSearchParams(location.search).has("profile");
const renderSamples = [];
let pollTimer = null, openedLocalFile = false, cameraFrame = null;
const inspector = createMemoryInspector($("inspection"), {
  onLinks: links => { state.links = links; scheduleDraw(); },
  onClose: () => { $("map").classList.remove("inspection-open"); $("inspect-memory").setAttribute("aria-pressed", "false"); },
  onFocus: links => {
    if (!links.length) return;
    stopCamera();
    const panel = $("inspection").getBoundingClientRect(), bounds = canvas.getBoundingClientRect();
    const rect = [24, 115, Math.max(80, panel.left - bounds.left - 48), Math.max(80, state.height - 205)];
    Object.assign(state, fitPoints([projectedPoint(links[0].memory), ...links.map(item => projectedPoint(item.p))], { ...state, rect }));
    scheduleDraw();
  },
});

function sampledOverlay(system) {
  const histories = state.history === null ? state.data.histories : [state.data.histories[state.history]];
  return histories.some(history => (history.memories[system.id]?.length || 0) < (history.memoryTotals?.[system.id] || 0));
}

function inspectMemory(item = null) {
  if (!state.data) return;
  if (!state.system) {
    state.system = state.data.systems[0]?.id || ""; $("memory-select").value = state.system;
    if (!state.system) { announce("This atlas contains no memory overlays."); return; }
    rebuild();
  }
  let history = state.history;
  if (history === null && item?.kind === "memory") history = state.data.histories.findIndex(candidate => candidate.memories[state.system]?.includes(item.p));
  if (history === null && item?.kind === "turn") history = state.data.sessions[item.session].histories[0];
  if (history === null || history === undefined || history < 0) history = inspector.context()?.history ?? 0;
  $("map").classList.add("inspection-open"); $("inspect-memory").setAttribute("aria-pressed", "true");
  inspector.open({ history, system: state.system, memory: item?.kind === "memory" ? item.p : undefined, turn: item?.kind === "turn" ? item : undefined, localFile: openedLocalFile });
}

async function pollDataset() {
  if (openedLocalFile) return;
  try {
    const response = await fetch("./atlas.json", { cache: "no-store" });
    if (response.ok) {
      const data = await unpackAtlas(await response.json(), (loaded, total) => {
        $("loading-detail").textContent = `Loading public benchmark map: ${Math.round(100 * loaded / total)}%`;
      });
      if (!openedLocalFile) loadAtlas(data);
      return;
    }
    if (response.status !== 404) throw new Error(`Could not read atlas (${response.status}).`);
    const status = await fetch("./build-status.json", { cache: "no-store" });
    if (status.ok) {
      const progress = await status.json();
      $("loading-detail").textContent = progress.message || "Building the real-data atlas locally…";
    } else $("loading-detail").textContent = "Open a prepared atlas JSON, or run the local atlas builder. No synthetic data is loaded.";
  } catch (error) { $("loading-detail").textContent = error.message; }
  pollTimer = setTimeout(pollDataset, 3000);
}

function loadAtlas(raw) {
  const data = validateAtlas(raw);
  stopCamera(); state.example = null; state.beforeExample = null;
  $("example-bubble").hidden = true; $("map").classList.remove("reader-open");
  $("exit-example").hidden = true;
  state.data = data;
  state.tsneLayout = data.tsneDefault || (data.tsneLayouts?.original ? "original" : null);
  inspector.setData(data);
  if (state.method === "tsne" && !data.projections?.tsne) state.method = "umap";
  if (!data.projections3d?.[state.method]) state.dimension = 2;
  state.history = null; state.system = ""; state.search = ""; state.selected = null; state.selectedPoint = null; state.hover = null;
  $("dataset-name").textContent = `${data.title} · ${number(data.histories.length)} questions`;
  $("history-input").max = String(data.histories.length);
  $("history-total").textContent = `/ ${number(data.histories.length)}`;
  $("memory-select").replaceChildren(option("", "Memories off"), ...data.systems.map(system => option(system.id, `${system.label}${system.sampleLimit ? " · sample" : ""}`)));
  $("history-input").value = ""; $("memory-select").value = ""; $("search-input").value = "";
  $("model-label").textContent = `${data.embedding.model} · ${data.embedding.dimensions || ""}D`;
  $("model-detail").textContent = `${data.embedding.model}. ${data.embedding.note || "All points use one shared analysis embedder."}`;
  $("memory-detail").textContent = "Memory overlays are saved stores from the same benchmark histories. Large stores are uniformly sampled to the limit recorded in the file. A shared history is not verified source provenance.";
  $("loading").hidden = true;
  $("reader").hidden = true;
  state.examples = guidedExamples(data);
  $("examples-label").hidden = $("example-buttons").hidden = !state.examples.length;
  $("tsne-buttons").replaceChildren(...Object.entries(data.tsneLayouts || {}).sort(([a], [b]) => (a === "original") - (b === "original")).map(([id, settings]) => {
    const button = el("button", settings.label || id); button.dataset.tsneLayout = id;
    button.title = `Perplexity ${settings.perplexity}; ${settings.iterations} iterations; ${settings.metric} distance; late exaggeration ${settings.exaggeration ?? 1}. ${settings.preprocessing}. These settings alter visual structure, not the original similarities.`;
    button.addEventListener("click", () => { state.tsneLayout = id; setProjection("tsne"); });
    return button;
  }));
  $("examples").hidden = !state.examples.length;
  $("example-buttons").replaceChildren(...state.examples.map(example => {
    const button = el("button", example.label); button.dataset.example = example.id;
    button.setAttribute("aria-pressed", "false");
    button.addEventListener("click", () => { if (state.example?.id === example.id) exitExample(true); else openExample(example); });
    return button;
  }));
  setProjection(state.method);
  announce(`Loaded ${data.histories.length} histories and ${data.sessions.length} real conversations.`);
}

function setProjection(method) {
  if (!state.data) return;
  if (method === "tsne" && !state.data.projections?.tsne) return;
  if (state.dimension === 3 && !state.data.projections3d?.[method]) return;
  state.method = method;
  state.xy = projectionCoordinates(state.data, method, state.dimension, method === "tsne" ? state.tsneLayout : null);
  $("tsne-settings").hidden = method !== "tsne" || !Object.keys(state.data.tsneLayouts || {}).length;
  $("examples").hidden = !state.examples.length && $("tsne-settings").hidden;
  document.querySelectorAll("[data-tsne-layout]").forEach(button => button.setAttribute("aria-pressed", String(button.dataset.tsneLayout === state.tsneLayout)));
  state.moments = state.dimension === 3 ? coordinateMoments(state.xy) : null;
  $("depth-control").hidden = state.dimension !== 3;
  document.querySelectorAll("[data-projection]").forEach(button => {
    button.setAttribute("aria-pressed", String(button.dataset.projection === method));
    button.disabled = (state.dimension === 3 && !state.data.projections3d?.[button.dataset.projection]) || (button.dataset.projection === "tsne" && !state.data.projections?.tsne);
    button.title = button.dataset.projection === "tsne" && button.disabled ? (state.dimension === 3 ? "t-SNE is available in 2D. Use UMAP or PCA for 3D." : "Add a real t-SNE layout with --add-tsne. Reload when preprocessing finishes.") : "";
  });
  document.querySelectorAll("[data-dimension]").forEach(button => { button.setAttribute("aria-pressed", String(Number(button.dataset.dimension) === state.dimension)); button.disabled = Number(button.dataset.dimension) === 3 && (!spatialPaths || !Object.keys(state.data.projections3d || {}).length); button.title = button.disabled ? "3D needs real three-axis coordinates and WebGL. Build the atlas with --add-3d." : ""; });
  pathCanvas.hidden = state.dimension === 3; spatialCanvas.hidden = state.dimension !== 3;
  $("interaction-hint").textContent = state.dimension === 3 ? "Drag to rotate · Shift-drag to pan · Scroll to zoom · Click to read" : "Click to read · Click empty space to clear · Scroll to zoom · Drag to pan";
  canvas.setAttribute("aria-label", state.dimension === 3 ? "3D conversation paths. Drag to rotate, Shift-drag to pan, scroll to zoom, and select a turn to read." : "Conversation paths. Scroll to zoom, drag to pan, and select a turn to read its conversation.");
  const info = method === "tsne" && state.tsneLayout ? state.data.tsneLayouts[state.tsneLayout] : (state.dimension === 3 ? state.data.projections3d : state.data.projections)?.[method];
  $("projection-note").textContent = `${method === "tsne" ? `t-SNE${info?.label ? ` · ${info.label}` : ""}` : method.toUpperCase()} · ${state.dimension}D · one shared fit · paths follow turn order`;
  $("projection-detail").textContent = method === "umap"
    ? `${state.dimension}D UMAP uses cosine neighborhoods, ${info?.neighbors || 20} neighbors, and min_dist ${info?.minDist ?? .12}. ${info?.preprocessing || ""}. 2D and 3D UMAP are separately fitted layouts; both can distort distances and density.`
    : method === "tsne"
      ? `2D t-SNE fits all ${number(info.fitPoints)} points together using ${info.metric} neighborhoods, perplexity ${info.perplexity}, seed ${info.seed}, and ${info.iterations} optimization iterations after ${info.earlyExaggerationIterations} early-exaggeration iterations; late exaggeration ${info.exaggeration ?? 1}. ${info.preprocessing}. Exaggeration changes visual compactness. Apparent clusters, gaps, sizes, and global distances are not measures of memory quality.`
      : `PCA is the linear comparison view. Its first ${state.dimension} axes retain ${((info?.variance || 0) * 100).toFixed(1)}% of embedding variance. The 3D view uses a real third principal component, not invented depth.`;
  resetCamera(); rebuild();
  if (state.example) focusExample();
}

function setDimension(dimension) {
  if (!state.data || dimension === state.dimension || (dimension === 3 && !spatialPaths)) return;
  if (dimension === 3 && !Object.keys(state.data.projections3d || {}).length) return;
  state.dimension = dimension;
  setProjection(dimension === 3 && !state.data.projections3d[state.method] ? Object.keys(state.data.projections3d)[0] : state.method);
}

function projectedPoint(p) { return state.dimension === 3 ? rotatedXY(state.xy[p], state.rotation) : state.xy[p]; }

function updateRotation() {
  state.rotation = state.dimension === 3 ? orbitRotation(...state.orbit) : IDENTITY_ROTATION;
  state.pickRotation = null;
  state.pathPicking = null;
}

function rebuild() {
  if (!state.data) return;
  state.sessions = visibleSessions(state.data, state.history, state.search);
  state.memories = memoryPoints(state.data, state.system, state.history);
  state.buckets = new Map(); state.paths = []; state.pathKey = null;
  state.pickCoordinates = state.dimension === 3 ? state.xy.map(point => rotatedXY(point, state.rotation)) : state.xy;
  state.pathPicking = null;
  const turnPoints = new Set();
  const put = (p, item) => {
    const [x, y] = state.pickCoordinates[p], key = `${Math.floor(x * 128)},${Math.floor(y * 128)}`;
    if (!state.buckets.has(key)) state.buckets.set(key, []);
    state.buckets.get(key).push({ ...item, p });
  };
  let turns = 0;
  for (const id of state.sessions) {
    const conversation = state.data.sessions[id];
    const path = new Path2D();
    for (const [index, turn] of conversation.turns.entries()) {
      const [x, y] = state.pickCoordinates[turn.p];
      if (index === 0) path.moveTo(x, y); else path.lineTo(x, y);
      put(turn.p, { session: id, turn: index, kind: "turn" });
      turnPoints.add(turn.p);
      turns++;
    }
    state.paths.push(path);
  }
  for (const p of state.memories) put(p, { kind: "memory" });
  state.turnPoints = [...turnPoints];
  pointRenderer.setPoints(state.xy, state.turnPoints, []);
  memoryRenderer.setPoints(state.xy, [], state.memories);
  if (state.dimension === 3) spatialPaths.setPaths(state.xy, state.sessions.map(id => state.data.sessions[id]));
  state.pickRotation = state.rotation;
  $("map").dataset.conversationPaths = String(state.sessions.length);
  const system = state.data.systems.find(system => system.id === state.system);
  const sampled = system && sampledOverlay(system);
  $("counts").textContent = `${number(state.sessions.length)} conversation${state.sessions.length === 1 ? "" : "s"} · ${number(turns)} turns${state.memories.length ? ` · ${number(state.memories.length)} ${sampled ? "memory samples" : "memories"}` : ""}`;
  for (const store of state.data.systems) {
    const item = [...$("memory-select").options].find(option => option.value === store.id);
    if (item) item.textContent = `${store.label}${sampledOverlay(store) ? " · sample" : ""}`;
  }
  $("memory-key").classList.toggle("inactive", !system);
  $("memory-opacity").disabled = state.autoOpacity || !system;
  $("memory-layer-label").textContent = system ? `${system.label} memories${sampled ? " · sample" : ""}` : "Memories off";
  $("query-key").hidden = state.history === null;
  $("all-histories").setAttribute("aria-pressed", String(state.history === null));
  if ((state.selected !== null && !state.sessions.includes(state.selected)) || (state.selectedPoint !== null && state.selected === null && !state.memories.includes(state.selectedPoint))) clearSelection();
  state.hover = null; $("tooltip").hidden = true;
  if (inspector.isOpen()) {
    const context = inspector.context();
    if (context.system !== state.system || (state.history !== null && context.history !== state.history)) inspectMemory();
  }
  scheduleDraw();
}

function resize() {
  const bounds = canvas.getBoundingClientRect();
  state.width = bounds.width; state.height = bounds.height;
  state.baseScale = Math.max(1, Math.min(bounds.width - 70, bounds.height - 70) * .9);
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(bounds.width * ratio); canvas.height = Math.round(bounds.height * ratio);
  pointCanvas.width = canvas.width; pointCanvas.height = canvas.height;
  gridCanvas.width = canvas.width; gridCanvas.height = canvas.height;
  pathCanvas.width = canvas.width; pathCanvas.height = canvas.height;
  spatialCanvas.width = canvas.width; spatialCanvas.height = canvas.height; spatialPaths?.invalidate();
  state.pathKey = null;
  memoryCanvas.width = canvas.width; memoryCanvas.height = canvas.height;
  pointRenderer.invalidate(); memoryRenderer.invalidate();
  if (state.example && $("reader").hidden) focusExample(false);
  scheduleDraw();
}
function scheduleDraw() {
  if (state.pending) return;
  state.pending = true;
  requestAnimationFrame(() => { state.pending = false; draw(); });
}
function stopCamera() { if (cameraFrame !== null) cancelAnimationFrame(cameraFrame); cameraFrame = null; }
function resetCamera() { stopCamera(); state.zoom = 1; state.pan = [0, 0]; state.orbit = [-.45, .3]; updateRotation(); scheduleDraw(); }
function worldToScreen(p) {
  const [x, y] = projectedPoint(p), scale = state.baseScale * state.zoom;
  return [state.width / 2 + state.pan[0] + x * scale, state.height / 2 + state.pan[1] + y * scale];
}

function effectiveOpacity() {
  return state.autoOpacity
    ? { cloud: .9, memory: .95 }
    : { cloud: state.cloudOpacity, memory: state.memoryOpacity };
}

function draw() {
  const started = performance.now();
  const ratio = window.devicePixelRatio || 1;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, state.width, state.height);
  gridCtx.setTransform(ratio, 0, 0, ratio, 0, 0); gridCtx.clearRect(0, 0, state.width, state.height);
  if (!state.data) return;
  const scale = state.baseScale * state.zoom;
  const opacity = effectiveOpacity();
  $("cloud-opacity").disabled = state.autoOpacity;
  $("memory-opacity").disabled = state.autoOpacity || !state.system;
  const depth = state.dimension === 3 && state.depthShading ? depthCueForView(state.moments, state.rotation) : null;
  const view = { width: state.width, height: state.height, ratio, scale, pan: state.pan, rotation: state.rotation, depth, autoOpacity: state.autoOpacity, markerArea: state.markerArea, cloudOpacity: opacity.cloud, memoryOpacity: opacity.memory };
  pointRenderer.draw(view); memoryRenderer.draw(view);
  for (const [id, value] of [["cloud", state.autoOpacity ? pointRenderer.opacitySummary().cloud : opacity.cloud], ["memory", state.autoOpacity ? memoryRenderer.opacitySummary().memory : opacity.memory]]) {
    const percent = Math.round(value * 100);
    $(`${id}-opacity`).value = String(percent);
    $(`${id}-opacity-value`).textContent = `${state.autoOpacity ? "≈" : ""}${percent}%`;
    $(`${id}-opacity-value`).title = state.autoOpacity ? "Estimated mean opacity; each point adapts to local on-screen density." : "Uniform manual opacity";
  }
  const pointsDone = performance.now();
  gridCtx.strokeStyle = "#b9c6b9"; gridCtx.lineWidth = 1;
  if (state.dimension === 2) {
    for (const { position: x, opacity } of gridMarks(state.width, state.zoom, state.pan[0], state.baseScale)) { gridCtx.globalAlpha = .16 * opacity; gridCtx.beginPath(); gridCtx.moveTo(x, 0); gridCtx.lineTo(x, state.height); gridCtx.stroke(); }
    for (const { position: y, opacity } of gridMarks(state.height, state.zoom, state.pan[1], state.baseScale)) { gridCtx.globalAlpha = .16 * opacity; gridCtx.beginPath(); gridCtx.moveTo(0, y); gridCtx.lineTo(state.width, y); gridCtx.stroke(); }
  } else drawSpatialFrame(scale);
  const pathKey = [scale, ...state.pan, state.width, state.height, ratio].join(",");
  const pathWidth = contextPathWidth(state.sessions.length);
  if (state.dimension === 3) spatialPaths.draw({ ...view, pathAlpha: pathOpacity(state.sessions.length, state.zoom), pathWidth });
  else if (state.pathKey !== pathKey) {
    pathCtx.setTransform(ratio, 0, 0, ratio, 0, 0); pathCtx.clearRect(0, 0, state.width, state.height);
    pathCtx.save();
    pathCtx.translate(state.width / 2 + state.pan[0], state.height / 2 + state.pan[1]); pathCtx.scale(scale, scale);
    pathCtx.lineWidth = pathWidth / scale;
    pathCtx.strokeStyle = PLOT_STYLE.pathColor;
    pathCtx.globalAlpha = pathOpacity(state.sessions.length, state.zoom);
    for (const path of state.paths) pathCtx.stroke(path);
    pathCtx.restore(); state.pathKey = pathKey;
  }
  ctx.save();
  const pathsDone = performance.now();
  ctx.translate(state.width / 2 + state.pan[0], state.height / 2 + state.pan[1]); ctx.scale(scale, scale);
  ctx.globalAlpha = 1;
  const exampleSessions = new Set(state.example?.anchors.filter(anchor => anchor.kind === "turn").map(anchor => anchor.session) || []);
  if (state.hover?.session !== undefined && state.hover.session !== state.selected && !exampleSessions.has(state.hover.session)) drawConversation(state.hover.session, PLOT_STYLE.hoverLineWidth / scale, scale, PLOT_STYLE.hoverOpacity, PLOT_STYLE.hoverColor);
  for (const session of exampleSessions) drawConversation(session, PLOT_STYLE.selectedLineWidth / scale, scale, PLOT_STYLE.selectedOpacity, PLOT_STYLE.selectedColor);
  if (state.selected !== null && !inspector.isOpen()) drawConversation(state.selected, PLOT_STYLE.selectedLineWidth / scale, scale, PLOT_STYLE.selectedOpacity, PLOT_STYLE.selectedColor);
  const activeMatch = state.links.find(item => item.active);
  if (activeMatch && (inspector.isOpen() || activeMatch.session !== state.selected)) drawConversation(activeMatch.session, PLOT_STYLE.selectedLineWidth / scale, scale, PLOT_STYLE.selectedOpacity, PLOT_STYLE.selectedColor);
  if (state.links.length) {
    for (const link of state.links) {
      const source = projectedPoint(link.p), memory = projectedPoint(link.memory);
      ctx.globalAlpha = link.active ? .9 : .4; ctx.strokeStyle = ATLAS_COLORS.memory; ctx.lineWidth = (link.active ? 1.3 : .8) / scale; ctx.setLineDash([4 / scale, 4 / scale]);
      ctx.beginPath(); ctx.moveTo(...source); ctx.lineTo(...memory); ctx.stroke(); ctx.setLineDash([]);
      ctx.globalAlpha = 1; ctx.strokeStyle = link.active ? PLOT_STYLE.selectedColor : ATLAS_COLORS.turn; ctx.lineWidth = 1.3 / scale;
      ctx.beginPath(); ctx.arc(...source, (link.active ? 6 : 4) / scale, 0, 2 * Math.PI); ctx.stroke();
      ctx.strokeStyle = ATLAS_COLORS.memory; ctx.beginPath(); ctx.arc(...memory, 7 / scale, 0, 2 * Math.PI); ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }
  if (state.history !== null) {
    const p = state.data.histories[state.history].query, [x, y] = projectedPoint(p);
    const style = markerStyle("query", state.markerArea);
    ctx.fillStyle = ATLAS_COLORS.query;
    ctx.beginPath(); ctx.arc(x, y, style.diameter / 2 / scale, 0, 2 * Math.PI); ctx.fill();
  }
  if (state.selectedPoint !== null && !inspector.isOpen()) {
    const [x, y] = projectedPoint(state.selectedPoint); ctx.strokeStyle = state.selected === null ? ATLAS_COLORS.memory : PLOT_STYLE.selectedColor; ctx.lineWidth = 1.4 / scale; ctx.beginPath(); ctx.arc(x, y, 7 / scale, 0, 2 * Math.PI); ctx.stroke();
  }
  ctx.restore();
  if (state.example) drawExampleAnchors();
  $("zoom-level").textContent = `${Math.round(state.zoom * 100)}%`;
  if (profiling) {
    renderSamples.push({ total: performance.now() - started, points: pointsDone - started, paths: pathsDone - pointsDone });
    if (renderSamples.length > 60) renderSamples.shift();
    const summary = { frames: renderSamples.length };
    for (const key of ["total", "points", "paths"]) {
      const values = renderSamples.map(sample => sample[key]).sort((a, b) => a - b);
      summary[key] = { mean: values.reduce((a, b) => a + b, 0) / values.length, p95: values[Math.floor((values.length - 1) * .95)] };
    }
    $("map").dataset.profile = JSON.stringify(summary);
  }
}

function drawExampleAnchors() {
  for (const [index, anchor] of state.example.anchors.entries()) {
    const [x, y] = worldToScreen(anchor.p);
    const color = anchor.kind === "memory" ? ATLAS_COLORS.memory : PLOT_STYLE.selectedColor;
    ctx.globalAlpha = 1; ctx.strokeStyle = color; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(x, y, 7, 0, 2 * Math.PI); ctx.stroke();
    ctx.font = "11px Arial, sans-serif";
    const label = `${index + 1}`, labelWidth = ctx.measureText(label).width + 12;
    ctx.fillStyle = "#fff"; ctx.fillRect(x + 9, y - 19, labelWidth, 18);
    ctx.strokeStyle = "#ddd"; ctx.lineWidth = .5; ctx.strokeRect(x + 9, y - 19, labelWidth, 18);
    ctx.fillStyle = color; ctx.fillText(label, x + 15, y - 6);
  }
}

function drawSpatialFrame(scale) {
  const corners = [];
  for (let x = 0; x < 2; x++) for (let y = 0; y < 2; y++) for (let z = 0; z < 2; z++) corners.push([x - .5, y - .5, z - .5]);
  const screen = point => {
    const [x, y] = rotatedXY(point, state.rotation);
    return [state.width / 2 + state.pan[0] + x * scale, state.height / 2 + state.pan[1] + y * scale];
  };
  gridCtx.save(); gridCtx.globalAlpha = .16; gridCtx.strokeStyle = "#777"; gridCtx.lineWidth = .5;
  gridCtx.beginPath();
  for (let i = 0; i < 8; i++) for (const bit of [1, 2, 4]) if (i < (i ^ bit)) {
    gridCtx.moveTo(...screen(corners[i])); gridCtx.lineTo(...screen(corners[i ^ bit]));
  }
  gridCtx.stroke(); gridCtx.globalAlpha = .7; gridCtx.font = "10px Arial, sans-serif"; gridCtx.fillStyle = "#777";
  const origin = [state.width - 85, state.height - 125];
  for (let axis = 0; axis < 3; axis++) {
    const point = [0, 0, 0]; point[axis] = .035;
    const [x, y] = rotatedXY(point, state.rotation), end = [origin[0] + x * 850, origin[1] + y * 850];
    gridCtx.beginPath(); gridCtx.moveTo(...origin); gridCtx.lineTo(...end); gridCtx.stroke();
    gridCtx.fillText(`${state.method === "pca" ? "PC" : "U"}${axis + 1}`, end[0] + 3, end[1] - 3);
  }
  gridCtx.restore();
}

function openExample(example) {
  stopCamera(); clearTimeout(searchTimer);
  if (!state.example) state.beforeExample = { history: state.history, system: state.system, search: state.search, method: state.method, tsneLayout: state.tsneLayout, dimension: state.dimension, orbit: [...state.orbit], zoom: state.zoom, pan: [...state.pan] };
  clearSelection(); state.example = example;
  state.history = example.history; state.system = example.system; state.search = "";
  $("history-input").value = String(example.history + 1); $("memory-select").value = example.system; $("search-input").value = "";
  $("example-title").textContent = example.title;
  $("example-description").textContent = example.description;
  $("example-caveat").textContent = example.caveat;
  $("example-evidence").replaceChildren(...example.anchors.map((anchor, index) => {
    const button = el("button", "", `example-evidence ${anchor.kind}`);
    const date = anchor.kind === "turn" ? ` · ${(state.data.sessions[anchor.session].date || "Undated").split(" ")[0]}` : " · sample";
    button.append(el("span", `${index + 1} · ${anchor.label}${date}`), el("p", anchor.excerpt || state.data.points[anchor.p].text));
    button.title = "Read the full text";
    button.addEventListener("click", () => select(anchor));
    return button;
  }));
  $("example-bubble").hidden = false;
  $("exit-example").hidden = false;
  document.querySelectorAll("[data-example]").forEach(button => {
    const active = button.dataset.example === example.id;
    button.setAttribute("aria-pressed", String(active)); button.title = active ? "Click again to exit this example" : "Open guided example";
  });
  rebuild(); focusExample(); announce(example.title);
}

function exitExample(restore = false) {
  if (!state.example) return;
  stopCamera(); state.example = null; $("example-bubble").hidden = true;
  $("exit-example").hidden = true;
  document.querySelectorAll("[data-example]").forEach(button => { button.setAttribute("aria-pressed", "false"); button.title = "Open guided example"; });
  const previous = state.beforeExample; state.beforeExample = null;
  if (restore && previous) {
    clearSelection(); Object.assign(state, previous);
    $("history-input").value = previous.history === null ? "" : String(previous.history + 1);
    $("memory-select").value = previous.system; $("search-input").value = previous.search;
    setProjection(previous.method); Object.assign(state, { zoom: previous.zoom, pan: previous.pan, orbit: previous.orbit }); updateRotation(); scheduleDraw();
  }
}

function focusExample(animate = true) {
  if (!state.example || !state.width) return;
  const map = canvas.getBoundingClientRect(), bubble = $("example-bubble").getBoundingClientRect();
  const controls = document.querySelector(".map-layers").getBoundingClientRect();
  const top = controls.bottom - map.top + 22;
  const rect = state.width <= 480
    ? [24, top, state.width - 48, Math.max(70, bubble.top - map.top - top - 18)]
    : [24, top, Math.max(80, bubble.left - map.left - 48), Math.max(80, state.height - top - 90)];
  const target = fitPoints(state.example.anchors.map(anchor => projectedPoint(anchor.p)), { ...state, rect });
  stopCamera();
  if (!animate || matchMedia("(prefers-reduced-motion: reduce)").matches) { Object.assign(state, target); scheduleDraw(); return; }
  const initial = { zoom: state.zoom, pan: [...state.pan] }, started = performance.now();
  function frame(now) {
    const t = Math.min(1, (now - started) / 420), eased = t * t * (3 - 2 * t);
    state.zoom = Math.exp(Math.log(initial.zoom) + Math.log(target.zoom / initial.zoom) * eased);
    state.pan = initial.pan.map((value, index) => value + (target.pan[index] - value) * eased);
    scheduleDraw(); cameraFrame = t < 1 ? requestAnimationFrame(frame) : null;
  }
  cameraFrame = requestAnimationFrame(frame);
}

function drawConversation(id, width, scale, emphasis, color) {
  const conversation = state.data.sessions[id];
  ctx.globalAlpha = emphasis; ctx.beginPath();
  for (const [index, turn] of conversation.turns.entries()) { const [x, y] = projectedPoint(turn.p); if (index) ctx.lineTo(x, y); else ctx.moveTo(x, y); }
  ctx.strokeStyle = "rgba(255,255,255,.85)"; ctx.lineWidth = width + 2 / scale; ctx.stroke();
  ctx.strokeStyle = color; ctx.lineWidth = width;
  ctx.stroke();
  const style = markerStyle("turn", state.markerArea);
  for (const [index, turn] of conversation.turns.entries()) {
    const [x, y] = projectedPoint(turn.p); ctx.fillStyle = color;
    ctx.beginPath(); ctx.arc(x, y, style.diameter / 2 / scale, 0, 2 * Math.PI); ctx.fill();
    if (state.zoom > 2 || conversation.turns.length < 10) { ctx.font = `${10 / scale}px -apple-system, sans-serif`; ctx.fillText(String(index + 1), x + 5 / scale, y - 4 / scale); }
  }
  ctx.globalAlpha = 1;
}

function pointer(event) { const bounds = canvas.getBoundingClientRect(); return [event.clientX - bounds.left, event.clientY - bounds.top]; }
function hitTest(x, y) {
  if (!state.data) return null;
  if (state.pickRotation !== state.rotation) {
    state.buckets = new Map();
    state.pickCoordinates = state.dimension === 3 ? state.xy.map(point => rotatedXY(point, state.rotation)) : state.xy;
    state.pathPicking = null;
    const put = (p, item) => {
      const [x, y] = state.pickCoordinates[p], key = `${Math.floor(x * 128)},${Math.floor(y * 128)}`;
      if (!state.buckets.has(key)) state.buckets.set(key, []);
      state.buckets.get(key).push({ ...item, p });
    };
    for (const session of state.sessions) for (const [turn, item] of state.data.sessions[session].turns.entries()) put(item.p, { kind: "turn", session, turn });
    for (const p of state.memories) put(p, { kind: "memory" });
    state.pickRotation = state.rotation;
  }
  const scale = state.baseScale * state.zoom;
  const wx = (x - state.width / 2 - state.pan[0]) / scale, wy = (y - state.height / 2 - state.pan[1]) / scale;
  const radius = markerHitRadius(state.markerArea) / scale, bx = Math.floor(wx * 128), by = Math.floor(wy * 128), range = Math.ceil(radius * 128);
  let best = null, distance = radius * radius, memory = null, memoryDistance = radius * radius;
  for (let dx = -range; dx <= range; dx++) for (let dy = -range; dy <= range; dy++) {
    for (const item of state.buckets.get(`${bx + dx},${by + dy}`) || []) {
      if (!state.autoOpacity && ((item.kind === "turn" && state.cloudOpacity === 0) || (item.kind === "memory" && state.memoryOpacity === 0))) continue;
      const [px, py] = state.pickCoordinates[item.p], candidate = (wx - px) ** 2 + (wy - py) ** 2;
      if (candidate < distance) { best = item; distance = candidate; }
      if (item.kind === "memory" && candidate < memoryDistance) { memory = item; memoryDistance = candidate; }
    }
  }
  const memoryRadius = (markerStyle("memory", state.markerArea).size / 2 + 1) / scale;
  if (memory && memoryDistance <= memoryRadius * memoryRadius) return memory;
  if (best) return best;
  if (!state.pathPicking) state.pathPicking = buildPathIndex(state.pickCoordinates, state.sessions, state.data.sessions);
  return nearestPath(state.pathPicking, wx, wy, 5 / scale);
}

function clearSelection() {
  if (inspector.isOpen()) inspector.close();
  state.selected = null; state.selectedPoint = null; state.selectedTurn = null; state.hover = null;
  $("reader").hidden = true; $("tooltip").hidden = true; canvas.classList.remove("over-point");
  $("map").classList.remove("reader-open");
  scheduleDraw();
}

function showHover(item, x, y) {
  state.hover = item;
  canvas.classList.toggle("over-point", Boolean(item));
  const tooltip = $("tooltip"); tooltip.hidden = !item;
  if (item) {
    const text = state.data.points[item.p].text;
    tooltip.textContent = `${item.kind === "memory" ? "Stored memory" : `Conversation ${state.data.sessions[item.session].id} · Turn ${item.turn + 1}`}\n${text.slice(0, 150)}${text.length > 150 ? "…" : ""}`;
    tooltip.style.left = `${Math.max(8, Math.min(state.width - 285, x + 15))}px`;
    tooltip.style.top = `${Math.max(8, Math.min(state.height - 130, y + 15))}px`;
  }
  scheduleDraw();
}

function select(item) {
  if (!item) return;
  state.selectedPoint = item.p;
  state.selected = item.kind === "turn" ? item.session : null;
  state.selectedTurn = item.kind === "turn" ? item.turn : null;
  $("reader").hidden = false; $("tooltip").hidden = true;
  $("map").classList.add("reader-open");
  $("reader-kind").textContent = item.kind === "memory" ? "STORED MEMORY" : "CONVERSATION";
  $("focus-path").disabled = item.kind !== "turn";
  $("previous-turn").disabled = item.kind !== "turn"; $("next-turn").disabled = item.kind !== "turn";
  if (item.kind === "memory") {
    $("reader-title").textContent = state.data.systems.find(system => system.id === state.system)?.label || "Memory";
    $("reader-meta").textContent = "Saved memory sample · shared analysis embedding";
    const text = el("p", state.data.points[item.p].text); text.style.padding = "0 8px"; text.style.whiteSpace = "pre-wrap"; text.style.overflowWrap = "anywhere";
    $("reader-turns").replaceChildren(text);
  } else {
    const conversation = state.data.sessions[item.session];
    $("reader-title").textContent = `Conversation ${conversation.id}`;
    $("reader-meta").textContent = `${conversation.date} · ${conversation.turns.length} turns`;
    $("reader-turns").replaceChildren(...conversation.turns.map((turn, index) => {
      const node = el("button", "", `turn${index === item.turn ? " active" : ""}`);
      node.append(el("span", `${String(index + 1).padStart(2, "0")} · ${turn.role}`), el("p", state.data.points[turn.p].text));
      node.addEventListener("click", () => select({ kind: "turn", session: item.session, turn: index, p: turn.p }));
      return node;
    }));
  }
  announce(item.kind === "memory" ? "Selected a stored memory." : `Selected conversation ${state.data.sessions[item.session].id}.`);
  if (item.kind === "memory" || inspector.isOpen()) inspectMemory(item);
  scheduleDraw();
}

function focusPath() {
  if (state.selected === null) return;
  stopCamera();
  const points = state.data.sessions[state.selected].turns.map(turn => projectedPoint(turn.p));
  const map = canvas.getBoundingClientRect(), controls = document.querySelector(".map-layers").getBoundingClientRect();
  const top = controls.bottom - map.top + 24;
  const right = state.width > 480 && !$("reader").hidden ? $("reader").getBoundingClientRect().left - map.left - 24 : state.width - 24;
  Object.assign(state, fitPoints(points, { ...state, rect: [24, top, Math.max(80, right - 24), Math.max(80, state.height - top - 85)] }));
  scheduleDraw();
}
function zoom(factor, anchor = [0, 0]) { stopCamera(); Object.assign(state, zoomViewport(state, factor, anchor)); scheduleDraw(); }

let drag = null, hoverFrame = null;
canvas.addEventListener("pointerdown", event => {
  if (!state.data || (event.button !== 0 && !(state.dimension === 3 && event.button === 2))) return;
  stopCamera(); const [x, y] = pointer(event);
  drag = { x, y, pan: [...state.pan], orbit: [...state.orbit], rotate: state.dimension === 3 && !event.shiftKey && event.button === 0, moved: false };
  canvas.setPointerCapture(event.pointerId); canvas.classList.add("dragging");
});
canvas.addEventListener("pointermove", event => {
  const [x, y] = pointer(event);
  if (drag) {
    if (Math.hypot(x - drag.x, y - drag.y) > 3) drag.moved = true;
    if (drag.rotate && drag.moved) {
      state.orbit = [drag.orbit[0] + (x - drag.x) * .008, Math.max(-1.5, Math.min(1.5, drag.orbit[1] - (y - drag.y) * .008))]; updateRotation(); state.hover = null;
    } else if (!drag.rotate) state.pan = [drag.pan[0] + x - drag.x, drag.pan[1] + y - drag.y];
    $("tooltip").hidden = true; scheduleDraw(); return;
  }
  if (hoverFrame) cancelAnimationFrame(hoverFrame);
  hoverFrame = requestAnimationFrame(() => { hoverFrame = null; showHover(hitTest(x, y), x, y); });
});
canvas.addEventListener("pointerup", event => { if (drag && !drag.moved) { const hit = hitTest(...pointer(event)); if (hit) select(hit); else if (state.example) exitExample(true); else clearSelection(); } drag = null; canvas.classList.remove("dragging"); });
canvas.addEventListener("pointercancel", () => { drag = null; canvas.classList.remove("dragging"); });
canvas.addEventListener("contextmenu", event => { if (state.dimension === 3) event.preventDefault(); });
canvas.addEventListener("pointerleave", () => { if (!drag) showHover(null, 0, 0); });
canvas.addEventListener("wheel", event => { if (!state.data) return; event.preventDefault(); const [x, y] = pointer(event); zoom(wheelZoomFactor(event, state.height), [x - state.width / 2, y - state.height / 2]); }, { passive: false });
canvas.addEventListener("keydown", event => {
  if (!state.data) return;
  if (event.key === "+" || event.key === "=") { event.preventDefault(); zoom(1.25); }
  if (event.key === "-") { event.preventDefault(); zoom(.8); }
  if (["ArrowRight", "ArrowLeft", "Enter"].includes(event.key) && state.sessions.length) {
    event.preventDefault(); const current = state.sessions.indexOf(state.selected), direction = event.key === "ArrowLeft" ? -1 : 1;
    const id = state.sessions[(current + direction + state.sessions.length) % state.sessions.length];
    select({ kind: "turn", session: id, turn: 0, p: state.data.sessions[id].turns[0].p });
  }
});
document.addEventListener("keydown", event => {
  if (event.key !== "Escape" || $("about-dialog").open) return;
  if (state.example) { event.preventDefault(); exitExample(true); }
  else if (inspector.isOpen()) { event.preventDefault(); inspector.close(); }
  else if (!$("reader").hidden) { event.preventDefault(); clearSelection(); }
});
document.querySelectorAll("[data-projection]").forEach(button => button.addEventListener("click", () => setProjection(button.dataset.projection)));
document.querySelectorAll("[data-dimension]").forEach(button => button.addEventListener("click", () => setDimension(Number(button.dataset.dimension))));
function setHistory(index) {
  if (!state.data) return;
  const inspecting = inspector.isOpen();
  exitExample();
  state.history = index; $("history-input").value = index === null ? "" : String(index + 1);
  clearSelection(); resetCamera(); rebuild();
  if (inspecting) inspectMemory();
}
function applyHistoryInput() {
  if (!state.data) return;
  const input = $("history-input"), value = input.value;
  if (value === "") setHistory(null);
  else if (Number.isInteger(Number(value)) && Number(value) >= 1 && Number(value) <= state.data.histories.length) setHistory(Number(value) - 1);
  else input.value = state.history === null ? "" : String(state.history + 1);
}
$("history-input").addEventListener("change", applyHistoryInput);
$("history-input").addEventListener("keydown", event => { if (event.key === "Enter") { event.preventDefault(); applyHistoryInput(); event.target.blur(); } });
$("previous-history").addEventListener("click", () => { if (state.data) setHistory(stepHistory(state.history, state.data.histories.length, -1)); });
$("next-history").addEventListener("click", () => { if (state.data) setHistory(stepHistory(state.history, state.data.histories.length, 1)); });
$("random-history").addEventListener("click", () => { if (state.data) setHistory(randomHistory(state.history, state.data.histories.length)); });
$("all-histories").addEventListener("click", () => setHistory(null));
$("memory-select").addEventListener("change", event => { const inspecting = inspector.isOpen(); exitExample(); state.system = event.target.value; if (!state.system || state.selected === null) clearSelection(); rebuild(); if (inspecting && state.system) inspectMemory(); });
$("inspect-memory").addEventListener("click", () => { if (inspector.isOpen()) inspector.close(); else inspectMemory(state.selected !== null ? { kind: "turn", session: state.selected, turn: state.selectedTurn, p: state.selectedPoint } : state.selectedPoint !== null ? { kind: "memory", p: state.selectedPoint } : null); });
let searchTimer;
$("search-input").addEventListener("input", event => { clearTimeout(searchTimer); searchTimer = setTimeout(() => { exitExample(); state.search = event.target.value; rebuild(); }, 180); });
$("search-input").addEventListener("keydown", event => { if (event.key === "Enter") { clearTimeout(searchTimer); exitExample(); state.search = event.target.value; rebuild(); } });
$("reset-view").addEventListener("click", resetCamera);
$("zoom-in").addEventListener("click", () => zoom(1.25)); $("zoom-out").addEventListener("click", () => zoom(.8));
$("close-reader").addEventListener("click", clearSelection);
$("depth-shading").addEventListener("change", event => { state.depthShading = event.target.checked; scheduleDraw(); });
$("close-example").addEventListener("click", () => exitExample(true));
$("exit-example").addEventListener("click", () => exitExample(true));
$("refocus-example").addEventListener("click", () => focusExample());
$("auto-opacity").addEventListener("change", event => {
  const opacity = state.autoOpacity ? { cloud: pointRenderer.opacitySummary().cloud, memory: memoryRenderer.opacitySummary().memory } : effectiveOpacity();
  state.autoOpacity = event.target.checked;
  if (!state.autoOpacity) { state.cloudOpacity = Math.round(opacity.cloud * 100) / 100; state.memoryOpacity = Math.round(opacity.memory * 100) / 100; }
  scheduleDraw();
});
$("cloud-opacity").addEventListener("input", event => { state.cloudOpacity = Number(event.target.value) / 100; $("cloud-opacity-value").textContent = `${event.target.value}%`; scheduleDraw(); });
$("memory-opacity").addEventListener("input", event => { state.memoryOpacity = Number(event.target.value) / 100; $("memory-opacity-value").textContent = `${event.target.value}%`; scheduleDraw(); });
$("marker-size").addEventListener("input", event => { state.markerArea = Number(event.target.value); $("marker-size-value").textContent = `${event.target.value} pt²`; scheduleDraw(); });
pointCanvas.addEventListener("point-renderer-restored", scheduleDraw);
memoryCanvas.addEventListener("point-renderer-restored", scheduleDraw);
spatialCanvas.addEventListener("point-renderer-restored", scheduleDraw);
$("focus-path").addEventListener("click", focusPath);
function moveTurn(direction) { if (state.selected === null) return; const turns = state.data.sessions[state.selected].turns, index = (state.selectedTurn + direction + turns.length) % turns.length; select({ kind: "turn", session: state.selected, turn: index, p: turns[index].p }); $("reader-turns").querySelector(".active")?.scrollIntoView({ block: "nearest" }); }
$("previous-turn").addEventListener("click", () => moveTurn(-1)); $("next-turn").addEventListener("click", () => moveTurn(1));
$("about-button").addEventListener("click", () => $("about-dialog").showModal()); $("close-about").addEventListener("click", () => $("about-dialog").close());
$("open-file").addEventListener("click", () => $("file-input").click());
$("file-input").addEventListener("change", async event => {
  const file = event.target.files[0]; event.target.value = ""; if (!file) return;
  try { const data = JSON.parse(await file.text()); validateAtlas(data); openedLocalFile = true; clearTimeout(pollTimer); loadAtlas(data); }
  catch (error) { $("loading").hidden = false; $("loading-detail").textContent = error.message; }
});
new ResizeObserver(resize).observe($("map"));
pollDataset();
