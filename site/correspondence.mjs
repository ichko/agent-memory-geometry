/* Scores are precomputed from original embeddings, never projected coordinates. */
export function historyRows(data, history) {
  return data.histories[history].sessions.flatMap(session => data.sessions[session].turns.map((turn, index) => [session, index, turn.p]));
}

export function validateCorrespondence(matrix, data, history) {
  const fail = message => { throw new Error(message); };
  if (!data.correspondence || matrix?.schemaVersion !== 1 || matrix.fingerprint !== data.correspondence.fingerprint || matrix.history !== history || matrix.metric !== "cosine" || matrix.dimensions !== data.embedding.dimensions) fail("These similarities do not match this atlas. Rebuild them from its original vectors.");
  const expected = historyRows(data, history);
  if (!Array.isArray(matrix.rows) || matrix.rows.length !== expected.length || !matrix.rows.every((row, i) => Array.isArray(row) && row.length === 3 && row.every((value, j) => value === expected[i][j]))) fail("Similarity rows do not match this history's source turns.");
  if (!matrix.systems || typeof matrix.systems !== "object") fail("Missing memory similarity matrices.");
  const stores = data.histories[history].memories;
  for (const [system, memories] of Object.entries(stores)) {
    const block = matrix.systems[system];
    if (!block || !Array.isArray(block.memories) || block.memories.length !== memories.length || !block.memories.every((p, i) => p === memories[i])) fail("Similarity columns do not match this memory overlay.");
    const total = data.histories[history].memoryTotals?.[system] ?? memories.length;
    if (block.total !== total || !Array.isArray(block.scores) || block.scores.length !== expected.length * memories.length || !block.scores.every(value => Number.isFinite(value) && value >= -1 && value <= 1)) fail("Invalid cosine similarity matrix or store counts.");
  }
  return matrix;
}

export function nearestSourceTurns(matrix, system, column, limit = 5) {
  const block = matrix.systems[system];
  if (!block || !Number.isInteger(column) || column < 0 || column >= block.memories.length) return [];
  const count = block.memories.length;
  return matrix.rows.map(([session, turn, p], row) => ({ session, turn, p, row, memory: block.memories[column], score: block.scores[row * count + column] }))
    .sort((a, b) => b.score - a.score || a.row - b.row).slice(0, Math.max(0, limit));
}

export function bestMemoryColumn(matrix, system, row) {
  const block = matrix.systems[system];
  if (!block?.memories.length || !Number.isInteger(row) || row < 0 || row >= matrix.rows.length) return -1;
  let best = 0;
  for (let column = 1; column < block.memories.length; column++) if (block.scores[row * block.memories.length + column] > block.scores[row * block.memories.length + best]) best = column;
  return best;
}

export function heatmapColor(score) {
  // A fixed [-1, 1] scale shared by every history and store, not per-panel normalization.
  const t = Math.max(0, Math.min(1, Math.abs(score)));
  const low = [247, 251, 255], high = score < 0 ? [150, 150, 150] : [8, 48, 107];
  return low.map((value, i) => Math.round(value + (high[i] - value) * t));
}

export function heatmapCell(x, y, { left = 56, top = 28, width = 10, height = 8, rows, columns }) {
  const column = Math.floor((x - left) / width), row = Math.floor((y - top) / height);
  return row >= 0 && row < rows && column >= 0 && column < columns ? { row, column } : null;
}
