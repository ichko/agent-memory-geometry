import test from "node:test";
import assert from "node:assert/strict";
import { validateAtlas, projectionCoordinates, visibleSessions, memoryPoints, gridLines, gridMarks, stepHistory, randomHistory } from "../site/atlas.mjs";
import { wheelZoomFactor, zoomViewport, fitPoints } from "../site/navigation.mjs";
import { markerStyle, createPointRenderer, ATLAS_COLORS, PLOT_STYLE, markerHitRadius, pathOpacity, contextPathWidth, densityOpacityMap, densityOpacityAt } from "../site/points.mjs";
import { guidedExamples } from "../site/examples.mjs";
import { buildPathIndex, nearestPath } from "../site/paths.mjs";
import { historyRows, validateCorrespondence, nearestSourceTurns, bestMemoryColumn, heatmapColor, heatmapCell } from "../site/correspondence.mjs";
import { IDENTITY_ROTATION, orbitRotation, rotatedXY, createSpatialPaths, coordinateMoments, depthCueForView, depthWeight } from "../site/spatial.mjs";

// Minimal numeric fixtures test the viewer. They are never served as a demo.
test("markers use Matplotlib's default categorical colors", () => {
  assert.deepEqual(ATLAS_COLORS, { turn: "#1f77b4", memory: "#ff7f0e", query: "#d62728" });
  assert.deepEqual(markerStyle("turn").color, [31 / 255, 119 / 255, 180 / 255]);
  assert.deepEqual(markerStyle("memory").color, [1, 127 / 255, 14 / 255]);
});

test("dense-scatter points stay compact, borderless, and easy to click", () => {
  assert.equal(PLOT_STYLE.area, 5);
  assert.equal(PLOT_STYLE.edgeWidth, 0);
  for (const kind of ["turn", "memory", "query"]) {
    const style = markerStyle(kind);
    assert.ok(style.size < 6);
    assert.equal(style.edgeWidth, 0);
    assert.equal(style.shape, "circle");
    assert.ok(markerHitRadius() > style.size / 2);
  }
  assert.ok(markerStyle("memory").size > markerStyle("turn").size);
});

test("context paths are gray hairlines and do not become a dense ink layer", () => {
  assert.equal(PLOT_STYLE.pathColor, "#7f7f7f");
  assert.equal(PLOT_STYLE.lineWidth, .25);
  assert.ok(PLOT_STYLE.selectedLineWidth <= 2);
  assert.ok(PLOT_STYLE.selectedLineWidth > PLOT_STYLE.hoverLineWidth);
  assert.ok(PLOT_STYLE.hoverLineWidth > PLOT_STYLE.lineWidth);
  assert.equal(PLOT_STYLE.hoverColor, ATLAS_COLORS.turn);
  assert.equal(PLOT_STYLE.selectedColor, "#9467bd");
  assert.notEqual(PLOT_STYLE.selectedColor, PLOT_STYLE.hoverColor);
  assert.ok(PLOT_STYLE.selectedOpacity > PLOT_STYLE.hoverOpacity);
  assert.ok(PLOT_STYLE.hoverOpacity > pathOpacity(53));
  assert.ok(pathOpacity(18464) >= .014 && pathOpacity(18464) < .04);
  assert.ok(pathOpacity(18464) < pathOpacity(53));
  assert.ok(pathOpacity(18464, 40) > pathOpacity(18464, 1));
  for (const count of [1, 5, 53]) {
    for (const zoom of [.3, 1, 4, 40]) {
      assert.ok(pathOpacity(count, zoom) >= .35);
      assert.ok(pathOpacity(count, zoom) <= .45);
    }
    assert.ok(contextPathWidth(count) <= .5);
  }
  assert.ok(pathOpacity(53) >= .35);
  assert.ok(contextPathWidth(1) < PLOT_STYLE.hoverLineWidth);
  assert.ok(contextPathWidth(18464) < .26);
  assert.equal(pathOpacity(0), 0);
  assert.ok(pathOpacity(100, 1) > 0);
});

test("paths can be picked halfway between dots, with a narrow tolerance", () => {
  const xy = [[-.4, 0], [.4, 0]], sessions = [{ turns: [{ p: 0 }, { p: 1 }] }];
  const index = buildPathIndex(xy, [0], sessions);
  const hit = nearestPath(index, .1, .003, .005);
  assert.equal(hit.session, 0); assert.equal(hit.onPath, true);
  assert.equal(hit.p, 1); assert.equal(hit.turn, 1);
  assert.equal(nearestPath(index, .1, .02, .005), null);
});

test("path picking never invents a connection between sessions", () => {
  const xy = [[-.4, 0], [-.3, 0], [.3, 0], [.4, 0]];
  const sessions = [{ turns: [{ p: 0 }, { p: 1 }] }, { turns: [{ p: 2 }, { p: 3 }] }];
  const index = buildPathIndex(xy, [0, 1], sessions);
  assert.equal(nearestPath(index, 0, 0, .01), null);
  assert.equal(nearestPath(index, .35, 0, .01).session, 1);
  const filtered = buildPathIndex(xy, [0], sessions);
  assert.equal(nearestPath(filtered, .35, 0, .01), null);
});

test("path hit tests stay continuous across grid boundaries and rotation", () => {
  const xyz = [[-.4, -.39, -.2], [.4, .38, .2]], sessions = [{ turns: [{ p: 0 }, { p: 1 }] }];
  for (const rotation of [IDENTITY_ROTATION, orbitRotation(.7, -.4)]) {
    const xy = xyz.map(point => rotatedXY(point, rotation)), index = buildPathIndex(xy, [0], sessions);
    for (let i = 0; i < 100; i++) {
      const t = i / 99, x = xy[0][0] + (xy[1][0] - xy[0][0]) * t, y = xy[0][1] + (xy[1][1] - xy[0][1]) * t;
      assert.ok(nearestPath(index, x, y + .000001, .00002));
    }
  }
});

test("real 3D coordinates are required; a 2D atlas never gets invented depth", () => {
  const data = fixture();
  assert.throws(() => projectionCoordinates(data, "pca", 3), /No real 3D/);
  data.projections3d = { pca: { dimensions: 3 } };
  data.points.forEach((point, i) => { point.pca3d = [point.pca[0], point.pca[1], i * .1]; });
  validateAtlas(data);
  const xyz = projectionCoordinates(data, "pca", 3);
  assert.ok(xyz.every(point => point.length === 3 && point.every(Number.isFinite)));
  assert.notEqual(xyz[0][2], xyz[1][2]);
  data.points[0].pca3d = [0, 0];
  assert.throws(() => validateAtlas(data), /three finite coordinates/);
});

test("3D orbit rotates depth into the screen plane and preserves lengths", () => {
  assert.deepEqual(rotatedXY([2, 3, 4], IDENTITY_ROTATION), [2, 3]);
  const [x, y] = rotatedXY([0, 0, 1], orbitRotation(Math.PI / 2, 0));
  assert.ok(Math.abs(x - 1) < 1e-12); assert.equal(y, 0);
  const rotation = orbitRotation(.83, -.31), point = [.2, -.4, .3];
  const transformed = [0, 1, 2].map(row => point.reduce((sum, value, col) => sum + value * rotation[col * 3 + row], 0));
  assert.ok(Math.abs(Math.hypot(...point) - Math.hypot(...transformed)) < 1e-12);
});

test("depth shading follows camera depth, not screen position or relevance", () => {
  const points = [[0, 0, -1], [0, 0, 0], [0, 0, 1]], moments = coordinateMoments(points);
  const cue = depthCueForView(moments, IDENTITY_ROTATION);
  assert.equal(cue.center, 0);
  assert.ok(depthWeight(-1, cue) < depthWeight(0, cue));
  assert.ok(depthWeight(0, cue) < depthWeight(1, cue));
  assert.ok(depthWeight(-100, cue) > 0);
  assert.equal(depthWeight(1, null), 1);
  assert.equal(depthWeight(1, { ...cue, strength: 0 }), 1);
  const turn = orbitRotation(Math.PI, 0);
  const after = depthCueForView(moments, turn);
  assert.ok(depthWeight(turn[8], after) < depthWeight(-turn[8], after));
  assert.ok(Number.isFinite(depthWeight(0, depthCueForView(coordinateMoments([[0, 0, 0]]), IDENTITY_ROTATION))));
});

test("Canvas fallback shades 3D points without changing size or hue; 2D stays flat", () => {
  const fills = [], radii = [];
  const ctx = { setTransform() {}, clearRect() {}, beginPath() {}, arc(x, y, r) { radii.push(r); }, fill() { fills.push({ color: this.fillStyle, alpha: this.globalAlpha }); } };
  const renderer = createPointRenderer({ getContext: () => ctx });
  renderer.setPoints([[0, 0, -.5], [0, 0, .5]], [0, 1], []);
  const view = { width: 200, height: 200, ratio: 1, scale: 100, pan: [0, 0], markerArea: 5, cloudOpacity: .2 };
  renderer.draw(view);
  assert.deepEqual(fills.map(fill => fill.alpha), [.2, .2]);
  renderer.draw({ ...view, depth: { center: 0, span: 1, strength: 1 } });
  assert.ok(fills[2].alpha < fills[3].alpha);
  assert.ok(fills.every(fill => fill.color === ATLAS_COLORS.turn));
  assert.ok(radii.every(radius => radius === radii[0]));
});

test("3D background paths upload once, include every session, and never join sessions", () => {
  const uploads = [], draws = [];
  const gl = new Proxy({
    getShaderParameter: () => true, getProgramParameter: () => true, isContextLost: () => false,
    createShader: () => ({}), createProgram: () => ({}), createBuffer: () => ({}),
    getAttribLocation: () => 0, getUniformLocation: (_, name) => name,
    bufferData: (_, values) => uploads.push([...values]), drawArrays: (_, __, count) => draws.push(count),
  }, { get(target, key) { return key in target ? target[key] : /^[A-Z0-9_]+$/.test(key) ? key : () => {}; } });
  const renderer = createSpatialPaths({ width: 800, height: 600, getContext: () => gl, addEventListener() {} });
  const xyz = [[0, 0, 0], [1, 0, 0], [0, 1, 1], [1, 1, 1]];
  renderer.setPaths(xyz, [{ turns: [{ p: 0 }, { p: 1 }] }, { turns: [{ p: 2 }, { p: 3 }] }]);
  assert.deepEqual(uploads, [[...xyz[0], ...xyz[1], ...xyz[2], ...xyz[3]]]);
  const view = { width: 800, height: 600, ratio: 1, scale: 500, pan: [0, 0], rotation: IDENTITY_ROTATION, pathAlpha: .04 };
  renderer.draw(view); renderer.draw(view);
  assert.deepEqual(draws, [4]);
  renderer.draw({ ...view, rotation: orbitRotation(.3, .2) });
  assert.deepEqual(draws, [4, 4]); assert.equal(uploads.length, 1);
});

test("filtered conversations stay strong at every zoom without changing marker sizes", () => {
  const xy = Array.from({ length: 20 }, (_, i) => [(i - 10) / 30, (i % 4 - 2) / 10]);
  const ids = xy.map((_, i) => i), style = markerStyle("turn");
  for (const zoom of [.3, 1, 4, 16, 40]) {
    const map = densityOpacityMap(xy, ids, { width: 800, height: 600, scale: zoom * 300, pan: [0, 0] });
    assert.ok(map.mean >= .72);
    assert.ok(map.mean <= .91);
    assert.deepEqual(markerStyle("turn"), style);
  }
});

test("local contrast preserves isolated points without making dense overlaps opaque", () => {
  const xy = [[.7, 0], [0, 0]], ids = Array(20000).fill(1); ids[0] = 0;
  const map = densityOpacityMap(xy, ids, { width: 800, height: 600, scale: 300, pan: [0, 0] });
  const isolated = densityOpacityAt(map, 610, 300), dense = densityOpacityAt(map, 400, 300);
  assert.ok(isolated > .5);
  assert.ok(dense < .15);
  assert.ok(isolated > 4 * dense);
});

test("contrast follows visible density after panning, zooming, or rotation, not total loaded count", () => {
  const xy = [[0, 0, 0], [.4, 0, .2]], ids = Array(20000).fill(1); ids[0] = 0;
  const view = { width: 800, height: 600, scale: 3000, pan: [0, 0] };
  const map = densityOpacityMap(xy, ids, view);
  assert.ok(map.visible < 1200);
  assert.ok(densityOpacityAt(map, 400, 300) > .7);
  const panned = densityOpacityMap(xy, ids, { ...view, pan: [-1200, 0] });
  assert.ok(panned.visible > 1200);
  assert.ok(densityOpacityAt(panned, 400, 300) < .15);
  const rotated = densityOpacityMap([[0, 0, 1]], [0], { width: 800, height: 600, scale: 100, pan: [0, 0], rotation: orbitRotation(Math.PI / 2, 0) });
  assert.ok(rotated.mean > .8);
});

test("bilinear contrast stays continuous across pixel-cell boundaries", () => {
  const view = { width: 800, height: 600, scale: 100, pan: [0, 0] };
  const map = densityOpacityMap([[0, 0]], Array(2000).fill(0), view);
  for (let x = 350; x <= 450; x += 24) assert.ok(Math.abs(densityOpacityAt(map, x - .001, 300) - densityOpacityAt(map, x + .001, 300)) < .001);
});

test("guided camera fits anchors into the free plot area, not behind a bubble", () => {
  const points = [[-.1, -.15], [.1, .15]], view = { width: 1000, height: 700, baseScale: 500, rect: [24, 100, 600, 500] };
  const camera = fitPoints(points, view);
  for (const point of points) {
    const x = view.width / 2 + camera.pan[0] + point[0] * view.baseScale * camera.zoom;
    const y = view.height / 2 + camera.pan[1] + point[1] * view.baseScale * camera.zoom;
    assert.ok(x >= 52 && x <= 596);
    assert.ok(y >= 128 && y <= 572);
  }
  assert.deepEqual(fitPoints([], view), { zoom: 1, pan: [0, 0] });
  assert.ok(Number.isFinite(fitPoints([[0, 0], [0, 0]], view).zoom));
});

test("guided examples resolve actual source roles and sampled stores or stay absent", () => {
  const data = fixture();
  assert.deepEqual(guidedExamples(data), []);
  data.histories[0].question = "How much more miles per gallon compared to now?";
  data.histories[0].sessions = [0, 1];
  data.points[0].text = "My car was getting 30 miles per gallon.";
  data.points[2].text = "Now it gets 28 miles per gallon.";
  data.points[3].text = "User plans to get a tune-up for their car.";
  data.systems = [{ id: "m2", label: "Ext-Dense" }]; data.histories[0].memories = { m2: [3] };
  const examples = guidedExamples(data);
  assert.deepEqual(examples.map(example => example.id), ["lost-detail", "two-values"]);
  assert.deepEqual(examples[0].anchors.map(anchor => anchor.p), [0, 3]);
  assert.notEqual(examples[1].anchors[0].session, examples[1].anchors[1].session);
  data.points[3].text += " It gets 30 miles per gallon.";
  assert.deepEqual(guidedExamples(data).map(example => example.id), ["two-values"]);
  data.sessions[1].turns[0].role = "assistant";
  assert.deepEqual(guidedExamples(data), []);
});

test("assistant example requires an assistant source and matching saved recommendation", () => {
  const data = fixture();
  data.histories[0].question = "Which Italian restaurant in Rome was recommended?";
  data.points[1].text = "Trattoria Da Enzo al 29, followed by Roscioli.";
  data.points[3].text = "Assistant recommended Trattoria Da Enzo al 29.";
  data.systems = [{ id: "m2", label: "Ext-Dense" }]; data.histories[0].memories = { m2: [3] };
  const [example] = guidedExamples(data);
  assert.equal(example.id, "assistant-memory");
  assert.equal(example.anchors[0].turn, 1);
  assert.equal(example.anchors[1].kind, "memory");
  data.sessions[0].turns[1].role = "user";
  assert.deepEqual(guidedExamples(data), []);
});

test("marker rendering has a native Canvas fallback without a GPU", () => {
  const contexts = [];
  createPointRenderer({ getContext(kind) { contexts.push(kind); return {}; } });
  assert.deepEqual(contexts, ["2d"]);
});

test("GPU batching reuses native marker artwork and avoids per-point drawing calls", () => {
  const draws = [], fills = [], premultiplication = [], depthUniforms = [], lighting = [], adaptive = [], positionUploads = [];
  const gl = new Proxy({
    getShaderParameter: () => true, getProgramParameter: () => true, isContextLost: () => false,
    createShader: () => ({}), createProgram: () => ({}), createBuffer: () => ({}), createTexture: () => ({}),
    getAttribLocation: () => 0, getUniformLocation: (_, name) => name,
    drawArrays: (_, __, count) => draws.push(count),
    pixelStorei: (key, value) => premultiplication.push([key, value]),
    bufferData: (_, values) => positionUploads.push(values.length),
    uniform3f: (name, ...values) => { if (name === "depthCue") depthUniforms.push(values); },
    uniform1f: (name, value) => { if (name === "shading") lighting.push(value); if (name === "adaptive") adaptive.push(value); },
  }, { get(target, key) { return key in target ? target[key] : /^[A-Z0-9_]+$/.test(key) ? key : () => {}; } });
  const paint = { beginPath() {}, arc() {}, fill() { fills.push(this.fillStyle); } };
  const canvas = {
    dataset: {}, width: 800, height: 600, addEventListener() {},
    ownerDocument: { createElement: () => ({ getContext: () => paint }) },
    getContext: kind => kind === "webgl" ? gl : null,
  };
  const renderer = createPointRenderer(canvas);
  renderer.setPoints([[0, 0], [.1, .1]], Array(10000).fill(0), [1]);
  const view = { width: 800, height: 600, ratio: 1, scale: 500, pan: [0, 0] };
  renderer.draw(view); renderer.draw(view);
  assert.equal(canvas.dataset.renderer, "webgl-sprites");
  assert.deepEqual(draws, [10000, 1]);
  assert.deepEqual(fills, [ATLAS_COLORS.turn, ATLAS_COLORS.memory]);
  assert.deepEqual(premultiplication, [["UNPACK_PREMULTIPLY_ALPHA_WEBGL", true]]);
  renderer.draw({ ...view, pan: [20, 0] });
  assert.deepEqual(draws, [10000, 1, 10000, 1]);
  assert.deepEqual(depthUniforms.at(-1), [0, 1, 0]);
  assert.equal(lighting.at(-1), 0);
  renderer.draw({ ...view, depth: { center: .2, span: .4, strength: 1 } });
  assert.deepEqual(depthUniforms.at(-1), [.2, .4, 1]);
  assert.equal(lighting.at(-1), 1);
  assert.deepEqual(draws, [10000, 1, 10000, 1, 10000, 1]);
  assert.equal(fills.length, 2);
  const automatic = { ...view, autoOpacity: true, cloudOpacity: .9, memoryOpacity: .95 };
  renderer.draw(automatic); renderer.draw(automatic);
  assert.equal(adaptive.at(-1), 1);
  assert.deepEqual(draws, [10000, 1, 10000, 1, 10000, 1, 10000, 1]);
  assert.deepEqual(positionUploads, [30000, 3]);
  assert.ok(renderer.opacitySummary().cloud < .2);
  assert.ok(renderer.opacitySummary().memory > .8);
  assert.equal(fills.length, 2);
});

function fixture() {
  return {
    schemaVersion: 2, title: "Unit fixture", embedding: { model: "test-only", kind: "analysis" },
    points: [
      { text: "Turn one", umap: [-1, 1], pca: [0, 1] },
      { text: "Turn two", umap: [1, -1], pca: [1, 0] },
      { text: "Another conversation", umap: [0, 0], pca: [-1, 0] },
      { text: "Memory", umap: [.5, .5], pca: [.5, .5] },
    ],
    sessions: [
      { id: "c1", label: "First", turns: [{ p: 0, role: "user" }, { p: 1, role: "assistant" }], histories: [0] },
      { id: "c2", label: "Second", turns: [{ p: 2, role: "user" }], histories: [1] },
    ],
    histories: [
      { question: "First question", query: 0, sessions: [0], memories: { m1: [3] } },
      { question: "Second question", query: 1, sessions: [1], memories: { m1: [3] } },
    ],
    systems: [{ id: "m1", label: "Test store" }],
  };
}

test("validates both projection coordinates and real embedding provenance", () => {
  assert.equal(validateAtlas(fixture()).histories.length, 2);
  const data = fixture(); data.embedding.kind = "synthetic";
  assert.throws(() => validateAtlas(data), /real model embeddings/);
});
test("rejects dangling turns, unknown systems, and missing projections", () => {
  const data = fixture(); data.sessions[0].turns[0].p = 999;
  assert.throws(() => validateAtlas(data), /Invalid conversation turn/);
  const other = fixture(); other.histories[0].memories.unknown = [0];
  assert.throws(() => validateAtlas(other), /Unknown memory point or system/);
  const missing = fixture(); delete missing.points[0].umap;
  assert.throws(() => validateAtlas(missing), /finite umap/);
});
test("the default overview includes every conversation, not one history", () => {
  assert.deepEqual(visibleSessions(fixture()), [0, 1]);
  assert.deepEqual(visibleSessions(fixture(), 0), [0]);
});
test("search filters source text without changing conversation turn order", () => {
  const data = fixture();
  assert.deepEqual(visibleSessions(data, null, "ANOTHER"), [1]);
  assert.deepEqual(visibleSessions(data, 0, "another"), []);
  assert.deepEqual(data.sessions[0].turns.map(turn => turn.p), [0, 1]);
});
test("memory overlays deduplicate points and respect the selected history", () => {
  assert.deepEqual(memoryPoints(fixture(), ""), []);
  assert.deepEqual(memoryPoints(fixture(), "m1"), [3]);
  assert.deepEqual(memoryPoints(fixture(), "m1", 0), [3]);
});
test("both projections keep the same point identity and preserve aspect ratio", () => {
  const data = fixture(), umap = projectionCoordinates(data, "umap"), pca = projectionCoordinates(data, "pca");
  assert.equal(umap.length, data.points.length);
  assert.equal(pca.length, data.points.length);
  assert.deepEqual(umap[0], [-.5, -.5]);
  assert.notDeepEqual(umap, pca);
  assert.throws(() => projectionCoordinates(data, "random"), /Unknown projection/);
});

test("t-SNE is a genuine optional layout, not a renamed UMAP or fabricated third axis", () => {
  const data = fixture();
  assert.throws(() => projectionCoordinates(data, "tsne"), /No t-SNE/);
  data.projections = { tsne: { method: "t-SNE", fitPoints: 4 } };
  data.points.forEach((point, i) => { point.tsne = [i, i ** 2]; });
  validateAtlas(data);
  const positions = projectionCoordinates(data, "tsne");
  assert.equal(positions.length, data.points.length);
  assert.notDeepEqual(positions, projectionCoordinates(data, "umap"));
  assert.throws(() => projectionCoordinates(data, "tsne", 3), /No real 3D/);
  data.points[1].tsne = [NaN, 0];
  assert.throws(() => validateAtlas(data), /finite tsne/);
});

test("t-SNE comparison layouts preserve identities and never fabricate missing coordinates", () => {
  const data = fixture();
  data.projections = { tsne: { method: "t-SNE" } };
  data.tsneLayouts = { original: { pointKey: "tsne" }, local: { pointKey: "tsne_local" } }; data.tsneDefault = "local";
  data.points.forEach((point, i) => { point.tsne = [i, i * 2]; point.tsne_local = [i * 2, i ** 2]; });
  validateAtlas(data);
  assert.notDeepEqual(projectionCoordinates(data, "tsne", 2, "local"), projectionCoordinates(data, "tsne", 2, "original"));
  assert.equal(projectionCoordinates(data, "tsne", 2, "local").length, data.points.length);
  assert.throws(() => projectionCoordinates(data, "tsne", 2, "absent"), /Unknown t-SNE/);
  data.tsneDefault = "absent";
  assert.throws(() => validateAtlas(data), /default t-SNE/);
  data.tsneDefault = "local"; data.points[0].tsne_local = [0, NaN];
  assert.throws(() => validateAtlas(data), /comparison layout/);
});

function correspondenceFixture() {
  const data = fixture();
  data.embedding.dimensions = 3;
  data.correspondence = { fingerprint: "unit-fixture-only" };
  const matrix = { schemaVersion: 1, fingerprint: "unit-fixture-only", history: 0, metric: "cosine", dimensions: 3, rows: [[0, 0, 0], [0, 1, 1]], systems: { m1: { memories: [3], total: 1, scores: [.9, .2] } } };
  return { data, matrix };
}

test("correspondence preserves session and turn order and ranks original-space scores", () => {
  const { data, matrix } = correspondenceFixture();
  assert.deepEqual(historyRows(data, 0), [[0, 0, 0], [0, 1, 1]]);
  validateCorrespondence(matrix, data, 0);
  assert.deepEqual(nearestSourceTurns(matrix, "m1", 0).map(item => [item.p, item.memory, item.score]), [[0, 3, .9], [1, 3, .2]]);
  // Moving projection coordinates must have no effect on matching or ranking.
  data.points.forEach(point => { point.umap = [9999, -100]; });
  assert.equal(nearestSourceTurns(matrix, "m1", 0)[0].p, 0);
  assert.deepEqual(nearestSourceTurns(matrix, "m1", -1), []);
});

test("similarities from another atlas, history, row order, or memory sample are rejected", () => {
  for (const mutate of [
    matrix => { matrix.fingerprint = "stale"; },
    matrix => { matrix.history = 1; },
    matrix => { matrix.metric = "projected-distance"; },
    matrix => { matrix.dimensions = 2; },
    matrix => { matrix.rows.reverse(); },
    matrix => { matrix.systems.m1.memories = [2]; },
    matrix => { matrix.systems.m1.total = 99; },
    matrix => { matrix.systems.m1.scores = [NaN, .2]; },
    matrix => { matrix.systems.m1.scores = [1.01, .2]; },
    matrix => { matrix.systems.m1.scores.pop(); },
  ]) {
    const { data, matrix } = correspondenceFixture(); mutate(matrix);
    assert.throws(() => validateCorrespondence(matrix, data, 0));
  }
});

test("heatmap memory matching handles ties, empty stores, negative scores, and row-major order", () => {
  const { matrix } = correspondenceFixture();
  matrix.systems.m1 = { memories: [3, 2], total: 10, scores: [.1, .9, -.2, -.3] };
  assert.equal(bestMemoryColumn(matrix, "m1", 0), 1);
  assert.equal(bestMemoryColumn(matrix, "m1", 1), 0);
  assert.equal(nearestSourceTurns(matrix, "m1", 1)[0].score, .9);
  matrix.systems.m1.scores = [.5, .5, .5, .5];
  assert.equal(bestMemoryColumn(matrix, "m1", 0), 0);
  assert.equal(nearestSourceTurns(matrix, "m1", 0)[0].row, 0);
  matrix.systems.m1.memories = [];
  assert.equal(bestMemoryColumn(matrix, "m1", 0), -1);
  assert.deepEqual(nearestSourceTurns(matrix, "m1", 0), []);
});

test("heatmap colors have a fixed cosine scale and cell picking rejects labels and margins", () => {
  assert.deepEqual(heatmapColor(0), [247, 251, 255]);
  assert.deepEqual(heatmapColor(1), [8, 48, 107]);
  assert.deepEqual(heatmapColor(-1), [150, 150, 150]);
  assert.deepEqual(heatmapColor(2), heatmapColor(1));
  const geometry = { rows: 3, columns: 2 };
  assert.deepEqual(heatmapCell(57, 29, geometry), { row: 0, column: 0 });
  assert.deepEqual(heatmapCell(67, 45, geometry), { row: 2, column: 1 });
  assert.equal(heatmapCell(50, 29, geometry), null);
  assert.equal(heatmapCell(57, 0, geometry), null);
  assert.equal(heatmapCell(80, 29, geometry), null);
});
test("grid coordinates move with pan and zoom, and stay inside the viewport", () => {
  const initial = gridLines(800, 1, 0, 600);
  const panned = gridLines(800, 1, 17, 600);
  assert.ok(initial.includes(400));
  assert.ok(panned.includes(417));
  assert.notDeepEqual(initial, gridLines(800, 1.2, 0, 600));
  assert.ok(panned.every(x => x >= 0 && x <= 800));
});
test("wheel and trackpad zoom are proportional without a modifier", () => {
  assert.ok(wheelZoomFactor({ deltaY: -100 }) > 1);
  assert.ok(wheelZoomFactor({ deltaY: 100 }) < 1);
  assert.ok(wheelZoomFactor({ deltaY: -1 }) < wheelZoomFactor({ deltaY: -100 }));
  assert.equal(wheelZoomFactor({ deltaY: 0 }), 1);
  assert.equal(wheelZoomFactor({ deltaY: NaN }), 1);
  assert.ok(wheelZoomFactor({ deltaY: -1, ctrlKey: true }) > 1);
});
test("wheel delta units are normalized", () => {
  assert.equal(wheelZoomFactor({ deltaY: 1, deltaMode: 1 }), wheelZoomFactor({ deltaY: 16 }));
  assert.equal(wheelZoomFactor({ deltaY: 1, deltaMode: 2 }, 200), wheelZoomFactor({ deltaY: 200 }));
});
test("zoom preserves the point under the cursor after panning", () => {
  const before = { zoom: 1.5, pan: [30, -10] }, anchor = [100, 70];
  const world = anchor.map((value, i) => (value - before.pan[i]) / before.zoom);
  const after = zoomViewport(before, 1.3, anchor);
  for (let i = 0; i < 2; i++) assert.ok(Math.abs(world[i] * after.zoom + after.pan[i] - anchor[i]) < 1e-9);
});
test("zoom limits do not cause cursor drift", () => {
  assert.deepEqual(zoomViewport({ zoom: 40, pan: [10, 20] }, 2, [50, 60]), { zoom: 40, pan: [10, 20] });
  assert.deepEqual(zoomViewport({ zoom: .3, pan: [10, 20] }, .5, [50, 60]), { zoom: .3, pan: [10, 20] });
});

test("history arrows wrap, and random excludes the current history", () => {
  assert.equal(stepHistory(null, 500, 1), 0);
  assert.equal(stepHistory(null, 500, -1), 499);
  assert.equal(stepHistory(499, 500, 1), 0);
  assert.equal(stepHistory(0, 500, -1), 499);
  assert.equal(randomHistory(5, 10, () => .5), 4);
  assert.equal(randomHistory(5, 10, () => .99), 9);
  assert.equal(randomHistory(null, 10, () => 0), 0);
  assert.equal(randomHistory(0, 1), 0);
  assert.equal(randomHistory(null, 0), null);
});

test("grid detail fades in continuously instead of changing scale abruptly", () => {
  const before = gridMarks(800, 1 - 1e-7, 0, 640);
  const after = gridMarks(800, 1 + 1e-7, 0, 640);
  const closest = (marks, position) => marks.reduce((best, mark) => Math.abs(mark.position - position) < Math.abs(best.position - position) ? mark : best);
  for (const mark of before.filter(mark => mark.opacity > .99 && mark.position > 1 && mark.position < 799)) {
    const corresponding = closest(after, mark.position);
    assert.ok(Math.abs(mark.position - corresponding.position) < .001);
    assert.ok(Math.abs(mark.opacity - corresponding.opacity) < .001);
  }
  assert.ok(before.every(mark => mark.opacity >= 0 && mark.opacity <= 1));
});

test("markers have fixed screen-space sizes independent of zoom", () => {
  const sizes = [], positions = [];
  let radius;
  const ctx = {
    setTransform() {}, clearRect() {}, beginPath() {}, fill() {}, stroke() {}, closePath() {}, moveTo() {}, lineTo() {},
    arc(x, y, r) { positions.push([x, y]); radius = r; sizes.push(r); },
  };
  const renderer = createPointRenderer({ getContext: kind => kind === "webgl" ? null : ctx });
  renderer.setPoints([[.01, .01]], [0], []);
  const view = { width: 800, height: 600, ratio: 2, scale: 500, pan: [0, 0], markerArea: 5 };
  renderer.draw(view); renderer.draw({ ...view, scale: 5000 });
  assert.equal(sizes[0], sizes[1]);
  assert.equal(radius * 2, markerStyle("turn").diameter);
  assert.notDeepEqual(positions[0], positions[1]);
  assert.ok(markerStyle("memory").size > markerStyle("turn").size);
  assert.equal(markerStyle("memory").shape, "circle");
  assert.equal(markerStyle("turn").shape, "circle");
});

test("conversation and memory opacity apply independently to their markers", () => {
  const fills = [];
  const ctx = { setTransform() {}, clearRect() {}, beginPath() {}, arc() {}, stroke() {}, fill() { fills.push(this.globalAlpha); } };
  const renderer = createPointRenderer({ getContext: kind => kind === "webgl" ? null : ctx });
  renderer.setPoints([[0, 0]], [0, 0], [0]);
  const view = { width: 800, height: 600, ratio: 2, scale: 500, pan: [0, 0], cloudOpacity: .25, memoryOpacity: .7 };
  renderer.draw(view);
  assert.deepEqual(fills, [.25, .25, .7]);
  fills.length = 0;
  renderer.draw({ ...view, memoryOpacity: 0 });
  assert.deepEqual(fills, [.25, .25]);
  fills.length = 0;
  const { cloudOpacity, memoryOpacity, ...defaults } = view;
  renderer.draw(defaults);
  assert.deepEqual(fills, [PLOT_STYLE.cloudOpacity, PLOT_STYLE.cloudOpacity, PLOT_STYLE.memoryOpacity]);
});
