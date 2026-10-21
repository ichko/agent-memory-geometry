/* Segment picking in the normalized plot plane. Never join separate sessions. */
const cellKey = (x, y) => x * 65536 + y;

export function buildPathIndex(xy, ids, sessions, cells = 32) {
  const buckets = new Map();
  for (const session of ids) {
    const turns = sessions[session].turns;
    for (let i = 1; i < turns.length; i++) {
      const p1 = turns[i - 1].p, p2 = turns[i].p;
      const [ax, ay] = xy[p1], [ex, ey] = xy[p2], dx = ex - ax, dy = ey - ay;
      const length2 = dx * dx + dy * dy;
      if (length2 < 1e-18) continue;
      const segment = { session, turn: i - 1, p1, p2, ax, ay, dx, dy, length2 };
      let x = Math.floor(ax * cells), y = Math.floor(ay * cells);
      const endX = Math.floor(ex * cells), endY = Math.floor(ey * cells);
      const stepX = Math.sign(dx), stepY = Math.sign(dy);
      const deltaX = dx ? 1 / Math.abs(dx * cells) : Infinity, deltaY = dy ? 1 / Math.abs(dy * cells) : Infinity;
      let nextX = dx ? ((x + (stepX > 0 ? 1 : 0)) / cells - ax) / dx : Infinity;
      let nextY = dy ? ((y + (stepY > 0 ? 1 : 0)) / cells - ay) / dy : Infinity;
      const limit = Math.abs(endX - x) + Math.abs(endY - y) + 2;
      for (let n = 0; n < limit; n++) {
        const key = cellKey(x, y);
        if (!buckets.has(key)) buckets.set(key, []);
        buckets.get(key).push(segment);
        if (x === endX && y === endY) break;
        if (nextX < nextY) { x += stepX; nextX += deltaX; }
        else if (nextY < nextX) { y += stepY; nextY += deltaY; }
        else { x += stepX; y += stepY; nextX += deltaX; nextY += deltaY; }
      }
    }
  }
  return { buckets, cells };
}

export function nearestPath(index, x, y, radius) {
  let best = null, distance = radius * radius;
  const seen = new Set(), { cells, buckets } = index;
  for (let bx = Math.floor((x - radius) * cells); bx <= Math.floor((x + radius) * cells); bx++) {
    for (let by = Math.floor((y - radius) * cells); by <= Math.floor((y + radius) * cells); by++) {
      for (const segment of buckets.get(cellKey(bx, by)) || []) {
        if (seen.has(segment)) continue;
        seen.add(segment);
        const { ax, ay, dx, dy, length2 } = segment;
        const t = Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / length2));
        const d2 = (x - ax - t * dx) ** 2 + (y - ay - t * dy) ** 2;
        if (d2 < distance) {
          distance = d2;
          best = { kind: "turn", session: segment.session, turn: segment.turn + (t > .5 ? 1 : 0), p: t > .5 ? segment.p2 : segment.p1, onPath: true };
        }
      }
    }
  }
  return best;
}
