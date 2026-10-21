import { PREVIEW } from "./preview-data.mjs";
import { preparePreview, previewProjection, previewDrag, previewShouldAnimate, previewAutoZoom, previewMemoryHighlight, previewFrontMemory, previewBubblePosition, previewAutoOrbit } from "./preview.mjs";

const stage = document.querySelector(".preview-stage");
const canvas = document.querySelector("#hero-canvas"), ctx = canvas.getContext("2d");
const bubble = document.querySelector("#preview-memory"), bubbleText = bubble.querySelector(".preview-memory-text");
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
const DEFAULT_ZOOM = 1.5;
const view = { yaw: -.45, pitch: .3, pan: [0, 0], zoom: DEFAULT_ZOOM };
let layout, width = 1, height = 1, ratio = 1, dragging = null, frame = null, last = 0, motionElapsed = 0;
let spinning = !reducedMotion.matches, visible = false, hovering = false, lastDraw = 0;
let activeMemory = null, highlightStarted = 0, bubbleHeight = 120;

function selectMemory(positions) {
  const expired = motionElapsed - highlightStarted >= 14000;
  const candidate = previewFrontMemory(PREVIEW.memoryDetails, positions, width, height, expired ? activeMemory?.point : null);
  const p = activeMemory && positions[activeMemory.point];
  const staysInFront = p && candidate && p[0] >= 30 && p[0] <= width - 25 && p[1] >= 22 && p[1] <= height - 22 && p[2] >= positions[candidate.point][2] - .25;
  if (candidate && (expired || !staysInFront)) {
    activeMemory = candidate; highlightStarted = motionElapsed;
    bubbleText.textContent = candidate.text;
    bubble.dataset.memoryId = candidate.point;
    bubble.hidden = false; bubbleHeight = bubble.offsetHeight;
  }
  if (!candidate) { activeMemory = null; bubble.hidden = true; }
  return activeMemory && previewMemoryHighlight([activeMemory.point], motionElapsed - highlightStarted);
}

function drawMemory(positions, highlight) {
  const [x, y] = positions[highlight.id];
  const paths = new Set(activeMemory.matches.map(m => m.path));
  ctx.globalAlpha = highlight.opacity * .8; ctx.strokeStyle = "#1f77b4"; ctx.lineWidth = .95;
  ctx.beginPath();
  for (const id of paths) PREVIEW.paths[id].forEach((p, i) => { const v = positions[p]; if (i) ctx.lineTo(v[0], v[1]); else ctx.moveTo(v[0], v[1]); });
  ctx.stroke();
  ctx.globalAlpha = highlight.opacity * .7; ctx.strokeStyle = "#ff7f0e"; ctx.lineWidth = .75; ctx.setLineDash([3, 3]);
  ctx.beginPath();
  for (const match of activeMemory.matches) { const p = positions[match.point]; ctx.moveTo(x, y); ctx.lineTo(p[0], p[1]); }
  ctx.stroke(); ctx.setLineDash([]);
  ctx.globalAlpha = highlight.opacity; ctx.fillStyle = "#1f77b4";
  for (const match of activeMemory.matches) { const p = positions[match.point]; ctx.beginPath(); ctx.arc(p[0], p[1], 2.3, 0, Math.PI * 2); ctx.fill(); }
  ctx.globalAlpha = highlight.opacity * .9; ctx.strokeStyle = "#ff7f0e"; ctx.lineWidth = 1.1;
  ctx.beginPath(); ctx.arc(x, y, highlight.radius, 0, Math.PI * 2); ctx.stroke();
  ctx.globalAlpha = highlight.opacity * .3; ctx.lineWidth = .6;
  ctx.beginPath(); ctx.arc(x, y, highlight.radius + 4, 0, Math.PI * 2); ctx.stroke();
  const box = stage.getBoundingClientRect();
  const bounds = { left: 12 - box.left, top: 12 - box.top, right: document.documentElement.clientWidth - box.left - 12, bottom: innerHeight - box.top - 12 };
  const blocked = [...document.querySelectorAll(".hero-copy > *")].map(el => { const r = el.getBoundingClientRect(); return { left: r.left - box.left - 8, right: r.right - box.left + 8, top: r.top - box.top - 8, bottom: r.bottom - box.top + 8 }; });
  const position = previewBubblePosition([x, y], bubble.offsetWidth, bubbleHeight, bounds, blocked);
  bubble.style.transform = `translate(${Math.round(position.x)}px,${Math.round(position.y)}px)`;
  bubble.style.opacity = highlight.opacity;
}

function updateMotion() {
  const rotating = spinning && !hovering && !dragging;
  canvas.setAttribute("aria-label", `3D UMAP of real LongMemEval turns and sampled memories. ${rotating ? "Auto-rotating." : "Rotation paused."} Drag to rotate. Scroll to zoom. Arrow keys move the view. Space toggles rotation. R resets.`);
}

function draw() {
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0); ctx.clearRect(0, 0, width, height);
  const positions = previewProjection(layout.coordinates, layout.dimensions, view.yaw, view.pitch, width, height, view.zoom * previewAutoZoom(motionElapsed), view.pan);
  const highlight = selectMemory(positions);
  ctx.lineWidth = .35; ctx.strokeStyle = "rgba(127,127,127,.24)";
  ctx.beginPath();
  for (const id of PREVIEW.backgroundPaths) {
    const path = PREVIEW.paths[id];
    path.forEach((id, i) => { const p = positions[id]; if (i) ctx.lineTo(p[0], p[1]); else ctx.moveTo(p[0], p[1]); });
  }
  ctx.stroke();
  const items = [...PREVIEW.turns.map(id => [id, false]), ...PREVIEW.memories.map(id => [id, true])];
  if (layout.dimensions === 3) items.sort((a, b) => positions[a[0]][2] - positions[b[0]][2]);
  for (const [id, memory] of items) {
    const [x, y, z] = positions[id], near = layout.dimensions === 3 ? Math.max(0, Math.min(1, (z + 1) / 2)) : .65;
    const size = memory ? 1.7 : .95;
    ctx.globalAlpha = (memory ? .72 : .52) * (.5 + .7 * near);
    ctx.fillStyle = memory ? "#ff7f0e" : "#1f77b4";
    ctx.beginPath(); ctx.arc(x, y, size, 0, Math.PI * 2); ctx.fill();
    if (layout.dimensions === 3 && near > .4) {
      ctx.globalAlpha *= .45; ctx.fillStyle = "#ffffff";
      ctx.beginPath(); ctx.arc(x - size * .24, y - size * .27, size * .28, 0, Math.PI * 2); ctx.fill();
    }
  }
  if (highlight) drawMemory(positions, highlight);
  ctx.globalAlpha = 1;
}

function tick(time) {
  frame = null;
  if (!previewShouldAnimate({ visible, hidden: document.hidden, spinning, dragging, hovering, dimensions: layout.dimensions })) { last = 0; return; }
  if (last) {
    const delta = Math.min(80, time - last), orbit = previewAutoOrbit(motionElapsed, delta);
    view.yaw += orbit.yaw; view.pitch = Math.max(-1.45, Math.min(1.45, view.pitch + orbit.pitch)); motionElapsed += delta;
  }
  last = time;
  if (time - lastDraw >= 32) { draw(); lastDraw = time; }
  frame = requestAnimationFrame(tick);
}

function animate() {
  if (frame === null && previewShouldAnimate({ visible, hidden: document.hidden, spinning, dragging, hovering, dimensions: layout.dimensions })) frame = requestAnimationFrame(tick);
}

function resetView() {
  motionElapsed = 0; activeMemory = null; highlightStarted = 0;
  Object.assign(view, { yaw: -.45, pitch: .3, pan: [0, 0], zoom: DEFAULT_ZOOM });
  draw();
}

function resize() {
  const box = stage.getBoundingClientRect(); width = box.width; height = box.height;
  ratio = Math.min(devicePixelRatio || 1, 2); canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio);
  draw();
}

function pauseForInput() {
  last = 0;
  if (frame !== null) { cancelAnimationFrame(frame); frame = null; }
  updateMotion();
}

function resumeAfterHover() {
  hovering = false; last = 0;
  updateMotion(); animate();
}

if (ctx && PREVIEW.layouts.umap3d) {
  layout = preparePreview(PREVIEW, "umap3d");
  updateMotion(); resize();
  document.querySelector("#preview-fallback").hidden = true; canvas.hidden = false;
  canvas.addEventListener("pointerenter", () => { hovering = true; pauseForInput(); draw(); });
  canvas.addEventListener("pointerleave", resumeAfterHover);
  canvas.addEventListener("pointerdown", event => {
    if (event.button !== 0 || dragging) return;
    dragging = { id: event.pointerId, x: event.clientX, y: event.clientY, origin: { ...view, pan: [...view.pan] } };
    pauseForInput(); canvas.focus(); canvas.setPointerCapture(event.pointerId);
    canvas.classList.add("is-dragging");
  });
  canvas.addEventListener("pointermove", event => {
    if (!dragging || event.pointerId !== dragging.id) return;
    Object.assign(view, previewDrag(layout.dimensions, dragging.origin, event.clientX - dragging.x, event.clientY - dragging.y)); draw();
  });
  const release = event => {
    if (dragging?.id !== event.pointerId) return;
    dragging = null; canvas.classList.remove("is-dragging");
    if (!hovering) resumeAfterHover();
  };
  canvas.addEventListener("pointerup", release); canvas.addEventListener("pointercancel", release); canvas.addEventListener("lostpointercapture", release);
  canvas.addEventListener("wheel", event => {
    event.preventDefault(); pauseForInput();
    const delta = event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? height : 1);
    view.zoom = Math.min(3, Math.max(.65, view.zoom * Math.exp(-delta * .0015))); draw(); animate();
  }, { passive: false });
  canvas.addEventListener("keydown", event => {
    if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", " ", "r", "R", "+", "-", "="].includes(event.key)) return;
    event.preventDefault();
    if (event.key === " " && layout.dimensions === 3) { spinning = !spinning; pauseForInput(); animate(); return; }
    pauseForInput();
    if (event.key.toLowerCase() === "r") { resetView(); animate(); return; }
    if (["+", "-", "="].includes(event.key)) view.zoom = Math.min(3, Math.max(.65, view.zoom * (event.key === "-" ? .9 : 1.1)));
    else Object.assign(view, previewDrag(layout.dimensions, view, event.key === "ArrowLeft" ? -12 : event.key === "ArrowRight" ? 12 : 0, event.key === "ArrowUp" ? -12 : event.key === "ArrowDown" ? 12 : 0));
    draw(); animate();
  });
  new ResizeObserver(resize).observe(stage);
  new IntersectionObserver(entries => { visible = entries[0].isIntersecting; if (!visible) last = 0; animate(); }).observe(stage);
  document.addEventListener("visibilitychange", () => { last = 0; animate(); });
  reducedMotion.addEventListener("change", event => { spinning = !event.matches; pauseForInput(); draw(); animate(); });
  // Start on first paint, rather than waiting for an observer or pointer event.
  const box = stage.getBoundingClientRect();
  visible = box.bottom > 0 && box.top < innerHeight && box.width > 0;
  animate();
}
