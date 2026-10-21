/* Same-origin static packs keep public data below GitHub's per-file limit. */
function localPath(path) {
  if (typeof path !== "string" || !/^(?:[a-zA-Z0-9_-]+\/)*[a-zA-Z0-9_.-]+\.json(?:\.gz)?$/.test(path)) throw new Error("Invalid demo data path.");
  return `./${path}`;
}

export async function fetchDataJson(path, fetcher = fetch) {
  const response = await fetcher(localPath(path));
  if (!response.ok) throw new Error(`Could not load demo data (${response.status}): ${path}`);
  if (!path.endsWith(".gz")) return response.json();
  if (/gzip/i.test(response.headers.get("content-encoding") || "")) return response.json();
  if (typeof DecompressionStream === "undefined") throw new Error("This demo needs a browser with gzip decompression support. Please update your browser.");
  return new Response(response.body.pipeThrough(new DecompressionStream("gzip"))).json();
}

export async function unpackAtlas(raw, onProgress = () => {}, fetcher = fetch) {
  if (raw?.format !== "memory-atlas-pack-v1") return raw;
  if (!Array.isArray(raw.parts) || !raw.parts.length || raw.parts.length > 100 || !Number.isInteger(raw.pointCount) || raw.pointCount < 1 || raw.pointCount > 1500000) throw new Error("Invalid demo data manifest.");
  for (const path of [raw.metadata, ...raw.parts]) localPath(path);
  const data = await fetchDataJson(raw.metadata, fetcher);
  data.points = [];
  // A small batch bounds simultaneous decompression while retaining point order.
  for (let start = 0; start < raw.parts.length; start += 3) {
    const blocks = await Promise.all(raw.parts.slice(start, start + 3).map(path => fetchDataJson(path, fetcher)));
    for (const block of blocks) {
      if (!Array.isArray(block) || data.points.length + block.length > raw.pointCount) throw new Error("Invalid demo point block.");
      for (const point of block) data.points.push(point);
    }
    onProgress(data.points.length, raw.pointCount);
  }
  if (data.points.length !== raw.pointCount) throw new Error("The demo data is incomplete.");
  return data;
}
