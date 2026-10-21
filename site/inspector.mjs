import { validateCorrespondence, nearestSourceTurns, bestMemoryColumn, heatmapColor, heatmapCell } from "./correspondence.mjs";

export function createMemoryInspector(root, callbacks) {
  const make = (tag, text = "", className = "") => { const node = document.createElement(tag); node.textContent = text; if (className) node.className = className; return node; };
  const button = (text, label, action) => { const node = make("button", text); node.type = "button"; if (label) node.setAttribute("aria-label", label); node.addEventListener("click", action); return node; };
  const heading = make("div", "", "inspection-heading");
  heading.append(make("h2", "Source ↔ memory"), button("×", "Close memory inspection", () => close()));
  const contextLabel = make("p", "", "inspection-context");
  const tabs = make("div", "", "inspection-tabs"); tabs.setAttribute("role", "group"); tabs.setAttribute("aria-label", "Memory inspection view");
  const connectionsTab = button("Connections", null, () => setMode("connections")); connectionsTab.dataset.inspectionMode = "connections";
  const heatmapTab = button("Heatmap", null, () => setMode("heatmap")); heatmapTab.dataset.inspectionMode = "heatmap";
  tabs.append(connectionsTab, heatmapTab);
  const status = make("p", "", "inspection-status"); status.setAttribute("role", "status");
  const fileInput = make("input"); fileInput.type = "file"; fileInput.accept = ".json,application/json"; fileInput.hidden = true;
  const importButton = button("Open similarities", null, () => fileInput.click()); importButton.className = "inspection-import"; importButton.hidden = true;
  const content = make("div", "", "inspection-content"); content.hidden = true;
  const navigation = make("div", "", "inspection-memory-nav");
  const memoryInput = make("input"); memoryInput.type = "number"; memoryInput.min = "1"; memoryInput.setAttribute("aria-label", "Memory number");
  const memoryCount = make("span");
  navigation.append(make("span", "Memory"), button("←", "Previous memory", () => chooseColumn(column - 1)), memoryInput, memoryCount, button("→", "Next memory", () => chooseColumn(column + 1)), button("Focus matches", null, () => callbacks.onFocus(links())));
  const heatmapView = make("section", "", "inspection-heatmap");
  const heatmapTitle = make("h3", "Conversation turns × stored memories");
  const heatmapHint = make("p", "Rows: source turns in session order. Columns: memories. Dark blue means higher cosine similarity; click a cell to compare the texts below.");
  const scroll = make("div", "", "heatmap-scroll"); scroll.tabIndex = 0; scroll.setAttribute("aria-label", "Scrollable turn-to-memory cosine similarity heatmap");
  const heatmapHeader = make("canvas", "", "heatmap-header"); heatmapHeader.setAttribute("aria-hidden", "true");
  const heatmap = make("canvas", "", "heatmap-matrix"); heatmap.setAttribute("role", "img");
  scroll.append(heatmapHeader, heatmap);
  const scale = make("div", "", "heatmap-scale");
  const scaleCanvas = make("canvas"); scaleCanvas.width = 160; scaleCanvas.height = 8;
  const scaleContext = scaleCanvas.getContext("2d");
  for (let i = 0; i < 160; i++) { scaleContext.fillStyle = `rgb(${heatmapColor(-1 + 2 * i / 159).join(",")})`; scaleContext.fillRect(i, 0, 1, 8); }
  scale.append(make("span", "Cosine"), make("span", "−1"), scaleCanvas, make("span", "+1 · fixed scale"));
  const hovered = make("p", "", "heatmap-readout"); hovered.setAttribute("aria-live", "polite");
  heatmapView.append(heatmapTitle, heatmapHint, scroll, scale, hovered);
  const pair = make("div", "", "inspection-pair");
  const sourceCard = make("section", "", "inspection-source"), memoryCard = make("section", "", "inspection-memory");
  const sourceLabel = make("h3"), memoryLabel = make("h3"), sourceText = make("p"), memoryText = make("p");
  sourceCard.append(sourceLabel, sourceText); memoryCard.append(memoryLabel, memoryText); pair.append(sourceCard, memoryCard);
  const scoreLabel = make("p", "", "inspection-score");
  const matchesTitle = make("h3", "Closest source turns · original embedding space");
  const matches = make("div", "", "inspection-matches");
  const caveat = make("p", "", "inspection-caveat");
  content.append(navigation, heatmapView, scoreLabel, pair, matchesTitle, matches, caveat);
  root.replaceChildren(heading, contextLabel, tabs, status, importButton, fileInput, content);
  let data = null, context = null, matrix = null, column = 0, row = 0, mode = "connections", generation = 0, cache = new Map(), geometry = null, baseImage = null;
  const block = () => matrix?.systems[context?.system];

  function setData(atlas) { generation++; data = atlas; cache.clear(); matrix = null; context = null; close(); }
  function close() { root.hidden = true; generation++; callbacks.onLinks([]); callbacks.onClose(); }
  function setMode(value) {
    mode = value;
    connectionsTab.setAttribute("aria-pressed", String(mode === "connections")); heatmapTab.setAttribute("aria-pressed", String(mode === "heatmap"));
    heatmapView.hidden = mode !== "heatmap"; matchesTitle.hidden = matches.hidden = mode === "heatmap";
    if (matrix && mode === "heatmap") drawHeatmap();
  }
  async function open(next, requestedMode) {
    root.hidden = false;
    if (requestedMode) setMode(requestedMode);
    const token = ++generation;
    context = next; matrix = null; content.hidden = true; importButton.hidden = true;
    const history = data.histories[next.history], store = data.systems.find(system => system.id === next.system);
    contextLabel.textContent = `History ${next.history + 1} · ${store?.label || "No memory store"}`;
    status.hidden = false; status.textContent = "Loading original-space similarities…";
    callbacks.onLinks([]);
    if (!data.correspondence) { status.textContent = "This atlas has no original-space similarities. Rebuild it with --correspondence; map distances cannot substitute for them."; return; }
    if (!history.memories[next.system]?.length) { status.textContent = "This history has no exported memories for the selected store. Choose another history or store."; return; }
    try {
      let loaded = cache.get(next.history);
      if (!loaded) {
        if (next.localFile) throw new Error("Open this history's similarity JSON from the atlas correspondence folder. Local files are not uploaded.");
        const response = await fetch(`./correspondence/h${next.history + 1}.json`, { cache: "no-store" });
        if (!response.ok) throw new Error("Similarities are unavailable. Run the atlas builder with --correspondence, or open the matching history JSON.");
        loaded = validateCorrespondence(await response.json(), data, next.history);
        cache.set(next.history, loaded);
        if (cache.size > 4) cache.delete(cache.keys().next().value);
      }
      if (token !== generation || root.hidden) return;
      display(loaded);
    } catch (error) { if (token === generation && !root.hidden) { status.textContent = error.message; importButton.hidden = false; } }
  }
  function display(loaded) {
    matrix = loaded; status.hidden = true; content.hidden = false; importButton.hidden = true;
    const memories = block().memories;
    const targetRow = context.turn ? matrix.rows.findIndex(([session, turn]) => session === context.turn.session && turn === context.turn.turn) : -1;
    column = Math.max(0, memories.indexOf(context.memory));
    if (targetRow >= 0) { row = targetRow; column = bestMemoryColumn(matrix, context.system, row); }
    else row = nearestSourceTurns(matrix, context.system, column, 1)[0]?.row ?? 0;
    geometry = { left: 56, top: 28, width: Math.max(9, Math.min(24, 384 / memories.length)), height: 9, rows: matrix.rows.length, columns: memories.length };
    baseImage = null;
    const sampled = memories.length < block().total;
    contextLabel.textContent += ` · ${memories.length.toLocaleString()}${sampled ? ` of ${block().total.toLocaleString()}` : ""} ${!sampled && memories.length === 1 ? "memory" : "memories"}${sampled ? " (sample)" : ""}`;
    caveat.textContent = `${sampled ? "Only sampled memories are shown. An unmatched turn does not establish absence from the full store. " : ""}Links are similarity matches, not verified provenance, factual coverage, or actual retrieval. Shared ${matrix.dimensions}D analysis embeddings; inputs truncated at ${data.embedding.maxTokens || "the model's"} tokens.`;
    memoryInput.max = String(memories.length); memoryCount.textContent = `/ ${memories.length}`;
    heatmap.setAttribute("aria-label", `${matrix.rows.length} source turns by ${memories.length} exported memories; original ${matrix.dimensions}-dimensional cosine similarity. Use the memory number and ranked matches for text inspection.`);
    update(); setMode(mode);
  }
  function links() {
    if (!matrix || !block()?.memories.length) return [];
    const result = nearestSourceTurns(matrix, context.system, column);
    const active = pairAt(row, column);
    if (!result.some(item => item.row === row)) result.push(active);
    return result.map(item => ({ ...item, active: item.row === row }));
  }
  function pairAt(selectedRow, selectedColumn) {
    const [session, turn, p] = matrix.rows[selectedRow];
    return { session, turn, p, row: selectedRow, memory: block().memories[selectedColumn], score: block().scores[selectedRow * block().memories.length + selectedColumn] };
  }
  function update() {
    const active = pairAt(row, column), conversation = data.sessions[active.session], turn = conversation.turns[active.turn];
    memoryInput.value = String(column + 1);
    memoryLabel.textContent = `M${column + 1} · stored memory`;
    sourceLabel.textContent = `${conversation.id} · turn ${active.turn + 1} · ${turn.role}`;
    sourceText.textContent = data.points[active.p].text; memoryText.textContent = data.points[active.memory].text;
    scoreLabel.textContent = `Cosine ${active.score.toFixed(3)} · original ${matrix.dimensions}D embeddings`;
    hovered.textContent = `${conversation.id} · T${active.turn + 1} × M${column + 1} · cosine ${active.score.toFixed(3)}`;
    matches.replaceChildren(...nearestSourceTurns(matrix, context.system, column).map((item, index) => {
      const match = button("", null, () => { row = item.row; update(); });
      match.className = `inspection-match${row === item.row ? " active" : ""}`;
      match.append(make("span", `${index + 1} · ${data.sessions[item.session].id} · turn ${item.turn + 1} · ${data.sessions[item.session].turns[item.turn].role}`), make("strong", item.score.toFixed(3)), make("p", data.points[item.p].text.slice(0, 160)));
      return match;
    }));
    callbacks.onLinks(links());
    if (mode === "heatmap") drawHeatmap();
  }
  function chooseColumn(value) {
    if (!matrix || !block()?.memories.length) return;
    column = ((value % block().memories.length) + block().memories.length) % block().memories.length;
    row = nearestSourceTurns(matrix, context.system, column, 1)[0].row;
    update();
  }
  function applyMemoryInput() { const value = Number(memoryInput.value); if (Number.isInteger(value) && value >= 1 && value <= block()?.memories.length) chooseColumn(value - 1); else memoryInput.value = String(column + 1); }
  memoryInput.addEventListener("change", applyMemoryInput);
  memoryInput.addEventListener("keydown", event => { if (event.key === "Enter") { event.preventDefault(); applyMemoryInput(); } });
  function drawHeatmap() {
    if (!matrix || !geometry) return;
    const { left, top, width, height, rows, columns } = geometry;
    const paint = heatmap.getContext("2d");
    if (!baseImage) {
      heatmap.width = Math.ceil(left + columns * width + 2); heatmap.height = top + rows * height + 2;
      paint.fillStyle = "#fff"; paint.fillRect(0, 0, heatmap.width, heatmap.height);
      const pixels = paint.createImageData(columns, rows);
      for (let i = 0; i < block().scores.length; i++) { pixels.data.set(heatmapColor(block().scores[i]), i * 4); pixels.data[i * 4 + 3] = 255; }
      const tiny = document.createElement("canvas"); tiny.width = columns; tiny.height = rows; tiny.getContext("2d").putImageData(pixels, 0, 0);
      paint.imageSmoothingEnabled = false; paint.drawImage(tiny, left, top, columns * width, rows * height);
      paint.font = "9px Arial, sans-serif"; paint.fillStyle = "#666";
      for (let c = 0; c < columns; c++) if (c % Math.max(1, Math.ceil(24 / width)) === 0) paint.fillText(`M${c + 1}`, left + c * width, 17);
      let previous = null;
      matrix.rows.forEach(([session, turn], r) => {
        if (session !== previous) { paint.fillStyle = "#666"; paint.fillText(data.sessions[session].id, 2, top + r * height + 8); paint.strokeStyle = "#fff"; paint.lineWidth = 1; paint.beginPath(); paint.moveTo(left, top + r * height); paint.lineTo(left + columns * width, top + r * height); paint.stroke(); previous = session; }
      });
      baseImage = paint.getImageData(0, 0, heatmap.width, heatmap.height);
      heatmapHeader.width = heatmap.width; heatmapHeader.height = top;
      heatmapHeader.getContext("2d").drawImage(heatmap, 0, 0, heatmap.width, top, 0, 0, heatmap.width, top);
    }
    paint.putImageData(baseImage, 0, 0);
    paint.strokeStyle = "#ff7f0e"; paint.lineWidth = 1.5; paint.strokeRect(left + column * width, top, width, rows * height);
    paint.strokeStyle = "#9467bd"; paint.lineWidth = 2; paint.strokeRect(left + column * width, top + row * height, width, height);
  }
  function cell(event) { const bounds = heatmap.getBoundingClientRect(); return heatmapCell((event.clientX - bounds.left) * heatmap.width / bounds.width, (event.clientY - bounds.top) * heatmap.height / bounds.height, geometry); }
  heatmap.addEventListener("pointermove", event => {
    if (!matrix || !geometry) return;
    const hit = cell(event); if (!hit) return;
    const item = pairAt(hit.row, hit.column);
    hovered.textContent = `${data.sessions[item.session].id} · T${item.turn + 1} × M${hit.column + 1} · cosine ${item.score.toFixed(3)}`;
    callbacks.onLinks([{ ...item, active: true }]);
  });
  heatmap.addEventListener("pointerleave", () => { if (matrix) callbacks.onLinks(links()); });
  heatmap.addEventListener("click", event => { const hit = geometry && cell(event); if (!hit) return; row = hit.row; column = hit.column; update(); });
  heatmap.addEventListener("keydown", event => { if (!matrix) return; const directions = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] }; const delta = directions[event.key]; if (!delta) return; event.preventDefault(); row = Math.max(0, Math.min(matrix.rows.length - 1, row + delta[0])); column = Math.max(0, Math.min(block().memories.length - 1, column + delta[1])); update(); });
  heatmap.tabIndex = 0;
  fileInput.addEventListener("change", async event => {
    const file = event.target.files[0], token = generation, history = context?.history; event.target.value = ""; if (!file || !context) return;
    try { const loaded = validateCorrespondence(JSON.parse(await file.text()), data, history); if (generation !== token || root.hidden) return; cache.set(history, loaded); display(loaded); }
    catch (error) { if (generation === token) { status.hidden = false; status.textContent = error.message; } }
  });
  setMode(mode);
  return { setData, open, close, isOpen: () => !root.hidden, context: () => context, clearLinks: () => callbacks.onLinks([]) };
}
