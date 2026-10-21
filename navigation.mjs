/* Smooth wheel/trackpad zoom, with a stable point under the cursor. */
export function wheelZoomFactor({ deltaY, deltaMode = 0, ctrlKey = false }, pageHeight = 345) {
  if (!Number.isFinite(deltaY)) return 1;
  const unit = deltaMode === 1 ? 16 : deltaMode === 2 ? pageHeight : 1;
  const pixels = Math.max(-240, Math.min(240, deltaY * unit));
  // Browser pinch gestures use ctrlKey and typically emit small pixel deltas.
  return Math.exp(-pixels * (ctrlKey ? .012 : .0025));
}

export function zoomViewport({ zoom, pan }, factor, anchor = [0, 0]) {
  const nextZoom = Math.max(.3, Math.min(40, zoom * factor));
  const ratio = nextZoom / zoom;
  return {
    zoom: nextZoom,
    pan: pan.map((value, i) => anchor[i] - (anchor[i] - value) * ratio),
  };
}

export function fitPoints(points, { width, height, baseScale, rect = [0, 0, width, height], padding = 28 }) {
  if (!points.length) return { zoom: 1, pan: [0, 0] };
  const xs = points.map(point => point[0]), ys = points.map(point => point[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  const [left, top, boxWidth, boxHeight] = rect;
  const scale = Math.min(Math.max(1, boxWidth - padding * 2) / Math.max(.018, maxX - minX), Math.max(1, boxHeight - padding * 2) / Math.max(.018, maxY - minY));
  const zoom = Math.max(.3, Math.min(30, scale / baseScale));
  return { zoom, pan: [left + boxWidth / 2 - width / 2 - (minX + maxX) / 2 * baseScale * zoom, top + boxHeight / 2 - height / 2 - (minY + maxY) / 2 * baseScale * zoom] };
}
