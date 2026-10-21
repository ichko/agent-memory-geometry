/* Native marker artwork, with optional 3D lighting and camera-depth haze. */
import { IDENTITY_ROTATION, rotatedXY, depthWeight } from "./spatial.mjs";
export const ATLAS_COLORS = Object.freeze({ turn: "#1f77b4", memory: "#ff7f0e", query: "#d62728" });
export const PLOT_STYLE = Object.freeze({ area: 5, cloudOpacity: .65, memoryOpacity: .8, edgeWidth: 0, lineWidth: .25, hoverLineWidth: 1.1, selectedLineWidth: 1.7, hoverOpacity: .9, selectedOpacity: 1, pathColor: "#7f7f7f", hoverColor: ATLAS_COLORS.turn, selectedColor: "#9467bd", pixelsPerPoint: 96 / 72 });
const rgb = Object.fromEntries(Object.entries(ATLAS_COLORS).map(([kind, hex]) => [kind, [1, 3, 5].map(start => parseInt(hex.slice(start, start + 2), 16) / 255)]));

export function markerStyle(kind, area = PLOT_STYLE.area) {
  const multiplier = kind === "memory" ? 1.8 : kind === "query" ? 3.2 : 1;
  const diameter = Math.sqrt(area * multiplier) * PLOT_STYLE.pixelsPerPoint;
  return { size: diameter, diameter, edgeWidth: 0, color: rgb[kind], shape: "circle" };
}

export function markerHitRadius(area = PLOT_STYLE.area) {
  return Math.max(9, markerStyle("turn", area).size / 2 + 4);
}

export function densityOpacityMap(coordinates, ids, { width, height, scale, pan, rotation = IDENTITY_ROTATION }, cap = .9, kind = "turn") {
  const cell = 24, columns = Math.ceil(width / cell) + 3, rows = Math.ceil(height / cell) + 3;
  const counts = new Float32Array(columns * rows), pixels = new Uint8Array(columns * rows * 4), samples = [];
  const limit = Math.min(8192, ids.length), weight = ids.length / Math.max(1, limit);
  let visible = 0;
  // Bilinear splats and texture filtering avoid abrupt opacity steps while panning.
  for (let i = 0; i < limit; i++) {
    const point = coordinates[ids[Math.floor(i * weight)]];
    const x = width / 2 + pan[0] + (rotation[0] * point[0] + rotation[3] * point[1] + rotation[6] * (point[2] || 0)) * scale;
    const y = height / 2 + pan[1] + (rotation[1] * point[0] + rotation[4] * point[1] + rotation[7] * (point[2] || 0)) * scale;
    if (x < -cell || y < -cell || x > width + cell || y > height + cell) continue;
    const gx = x / cell + 1, gy = y / cell + 1, ix = Math.floor(gx), iy = Math.floor(gy), tx = gx - ix, ty = gy - iy;
    counts[iy * columns + ix] += weight * (1 - tx) * (1 - ty);
    counts[iy * columns + ix + 1] += weight * tx * (1 - ty);
    counts[(iy + 1) * columns + ix] += weight * (1 - tx) * ty;
    counts[(iy + 1) * columns + ix + 1] += weight * tx * ty;
    if (x >= 0 && y >= 0 && x <= width && y <= height) { visible += weight; samples.push([x, y]); }
  }
  const sparse = 1 / (1 + (visible / 1800) ** 3);
  const floor = .035 + (cap * .8 - .035) * sparse;
  const target = kind === "memory" ? 3 : 5;
  for (let i = 0; i < counts.length; i++) pixels[i * 4 + 3] = Math.round(255 * (floor + (cap - floor) / Math.sqrt(1 + counts[i] / target)));
  const map = { pixels, columns, rows, cell, visible, mean: cap };
  if (samples.length) map.mean = samples.reduce((sum, [x, y]) => sum + densityOpacityAt(map, x, y), 0) / samples.length;
  return map;
}

export function densityOpacityAt(map, x, y) {
  const gx = Math.max(0, Math.min(map.columns - 1.001, x / map.cell + 1)), gy = Math.max(0, Math.min(map.rows - 1.001, y / map.cell + 1));
  const ix = Math.floor(gx), iy = Math.floor(gy), tx = gx - ix, ty = gy - iy;
  const alpha = (dx, dy) => map.pixels[((iy + dy) * map.columns + ix + dx) * 4 + 3] / 255;
  return (alpha(0, 0) * (1 - tx) + alpha(1, 0) * tx) * (1 - ty) + (alpha(0, 1) * (1 - tx) + alpha(1, 1) * tx) * ty;
}

export function pathOpacity(count, zoom = 1) {
  if (count <= 0) return 0;
  return Math.min(.45, Math.max(.014, .55 / (1 + (count / 120) ** 1.2), .9 * Math.max(1, zoom) ** .7 / Math.sqrt(count)));
}

export function contextPathWidth(count) {
  return PLOT_STYLE.lineWidth + .25 / (1 + Math.max(0, count) / 120);
}

export function createPointRenderer(canvas, { backend = "auto" } = {}) {
  if (backend !== "canvas" && canvas.ownerDocument) {
    const gl = canvas.getContext("webgl", { alpha: true, premultipliedAlpha: true, antialias: false, depth: false });
    if (gl) {
      if (canvas.dataset) canvas.dataset.renderer = "webgl-sprites";
      return createSpriteRenderer(canvas, gl);
    }
  }
  if (canvas.dataset) canvas.dataset.renderer = "canvas2d";
  const ctx = canvas.getContext("2d", { alpha: true });
  let coordinates = [], turns = [], memories = [], lastKey = null;
  const summary = { cloud: PLOT_STYLE.cloudOpacity, memory: PLOT_STYLE.memoryOpacity };
  const sprites = new Map();
  function sprite(kind, diameter, ratio) {
    if (!canvas.ownerDocument) return null;
    const key = [kind, diameter, ratio].join(",");
    if (sprites.has(key)) return sprites.get(key);
    const image = canvas.ownerDocument.createElement("canvas");
    image.width = image.height = Math.ceil((diameter + 2) * ratio);
    const paint = image.getContext("2d");
    paint.fillStyle = ATLAS_COLORS[kind];
    paint.beginPath(); paint.arc(image.width / 2, image.height / 2, diameter * ratio / 2, 0, 2 * Math.PI); paint.fill();
    if (sprites.size > 12) sprites.clear();
    sprites.set(key, image);
    return image;
  }
  return {
    opacitySummary: () => summary,
    invalidate() { lastKey = null; },
    setPoints(xy, turnIds, memoryIds) { coordinates = xy; turns = turnIds; memories = memoryIds; lastKey = null; },
    draw({ width, height, ratio, scale, pan, markerArea, rotation = IDENTITY_ROTATION, depth = null, autoOpacity = false, cloudOpacity = PLOT_STYLE.cloudOpacity, memoryOpacity = PLOT_STYLE.memoryOpacity }) {
      const key = [width, height, ratio, scale, ...pan, ...rotation, markerArea, autoOpacity, cloudOpacity, memoryOpacity, depth?.center, depth?.span, depth?.strength].join(",");
      if (key === lastKey) return;
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0); ctx.clearRect(0, 0, width, height);
      const centerX = width / 2 + pan[0], centerY = height / 2 + pan[1];
      for (const [ids, kind, alpha] of [[turns, "turn", cloudOpacity], [memories, "memory", memoryOpacity]]) {
        if (!ids.length || alpha === 0) continue;
        const density = autoOpacity ? densityOpacityMap(coordinates, ids, { width, height, scale, pan, rotation }, alpha, kind) : null;
        summary[kind === "turn" ? "cloud" : "memory"] = density?.mean ?? alpha;
        const { diameter } = markerStyle(kind, markerArea), r = diameter / 2;
        const image = sprite(kind, diameter, ratio);
        const side = image ? image.width / ratio : diameter;
        ctx.fillStyle = ATLAS_COLORS[kind]; ctx.globalAlpha = alpha;
        for (const p of ids) {
          const [rx, ry] = rotatedXY(coordinates[p], rotation);
          const x = centerX + rx * scale, y = centerY + ry * scale;
          if (x < -r || x > width + r || y < -r || y > height + r) continue;
          let opacity = density ? densityOpacityAt(density, x, y) : alpha;
          if (depth) {
            const point = coordinates[p];
            const z = rotation[2] * point[0] + rotation[5] * point[1] + rotation[8] * (point[2] || 0);
            opacity *= depthWeight(z, depth);
          }
          ctx.globalAlpha = Math.min(1, opacity);
          if (image) ctx.drawImage(image, x - side / 2, y - side / 2, side, side);
          else { ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI); ctx.fill(); }
        }
      }
      ctx.globalAlpha = 1; lastKey = key;
    },
  };
}

/* The GPU places native Canvas sprites and varies their alpha for 3D depth. */
function createSpriteRenderer(canvas, gl) {
  let coordinates = [], turns = [], memories = [], lastKey = null;
  const summary = { cloud: PLOT_STYLE.cloudOpacity, memory: PLOT_STYLE.memoryOpacity };
  const textures = new Map();
  const shader = (type, source) => {
    const result = gl.createShader(type); gl.shaderSource(result, source); gl.compileShader(result);
    if (!gl.getShaderParameter(result, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(result));
    return result;
  };
  const program = gl.createProgram();
  gl.attachShader(program, shader(gl.VERTEX_SHADER, `
    attribute vec3 position;
    uniform mat3 rotation;
    uniform vec2 viewport, pan;
    uniform float scale, size;
    uniform vec3 depthCue;
    varying mediump float depthWeight;
    varying mediump float cameraDepth;
    void main() {
      vec3 rotated = rotation * position;
      vec2 pixel = rotated.xy * scale + pan;
      gl_Position = vec4(2.0 * pixel.x / viewport.x, -2.0 * pixel.y / viewport.y, 0.0, 1.0);
      gl_PointSize = size;
      float t = clamp(0.5 + (rotated.z - depthCue.x) / depthCue.y, 0.0, 1.0);
      cameraDepth = t;
      depthWeight = mix(1.0, 0.55 + 0.9 * t * t, depthCue.z);
    }`));
  gl.attachShader(program, shader(gl.FRAGMENT_SHADER, `
    precision mediump float;
    uniform sampler2D sprite;
    uniform sampler2D density;
    uniform vec2 densityScale, densityOffset;
    uniform float adaptive;
    uniform float opacity;
    uniform float shading, fillRatio;
    varying mediump float depthWeight;
    varying mediump float cameraDepth;
    void main() {
      vec4 point = texture2D(sprite, gl_PointCoord);
      if (shading > 0.0 && point.a > 0.0) {
        vec2 p = (gl_PointCoord * 2.0 - 1.0) / fillRatio;
        vec3 normal = vec3(p.x, -p.y, sqrt(max(0.0, 1.0 - dot(p, p))));
        vec3 light = normalize(vec3(-0.45, 0.6, 1.0));
        float diffuse = 0.82 + 0.18 * max(0.0, dot(normal, light));
        float glint = 0.14 * pow(max(0.0, dot(normal, normalize(light + vec3(0.0, 0.0, 1.0)))), 18.0);
        float fog = 0.25 * (1.0 - cameraDepth) * (1.0 - cameraDepth);
        vec3 lit = point.rgb * diffuse + vec3(point.a * glint);
        point.rgb = mix(point.rgb, mix(min(lit, vec3(point.a)), vec3(point.a), fog), shading);
      }
      float alpha = mix(opacity, texture2D(density, gl_FragCoord.xy * densityScale + densityOffset).a, adaptive);
      gl_FragColor = point * min(1.0, alpha * depthWeight);
    }`));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
  gl.useProgram(program);
  const position = gl.getAttribLocation(program, "position"); gl.enableVertexAttribArray(position);
  const uniform = Object.fromEntries(["viewport", "pan", "scale", "size", "opacity", "sprite", "rotation", "depthCue", "shading", "fillRatio", "density", "densityScale", "densityOffset", "adaptive"].map(key => [key, gl.getUniformLocation(program, key)]));
  gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
  gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
  gl.activeTexture(gl.TEXTURE0); gl.uniform1i(uniform.sprite, 0);
  const densityTexture = gl.createTexture();
  gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, densityTexture); gl.uniform1i(uniform.density, 1);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 255]));
  gl.activeTexture(gl.TEXTURE0);
  const turnBuffer = gl.createBuffer(), memoryBuffer = gl.createBuffer();
  function upload(buffer, ids) {
    const data = new Float32Array(ids.length * 3);
    for (let i = 0; i < ids.length; i++) data.set(coordinates[ids[i]], i * 3);
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer); gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
  }
  function texture(kind, diameter, ratio) {
    const key = [kind, diameter, ratio].join(",");
    if (textures.has(key)) return textures.get(key);
    const image = canvas.ownerDocument.createElement("canvas");
    image.width = image.height = Math.ceil((diameter + 2) * ratio);
    const paint = image.getContext("2d"); paint.fillStyle = ATLAS_COLORS[kind];
    paint.beginPath(); paint.arc(image.width / 2, image.height / 2, diameter * ratio / 2, 0, 2 * Math.PI); paint.fill();
    const handle = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, handle);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
    if (textures.size >= 12) { for (const value of textures.values()) gl.deleteTexture(value.handle); textures.clear(); }
    const result = { handle, size: image.width }; textures.set(key, result); return result;
  }
  const api = {
    opacitySummary: () => summary,
    invalidate() { lastKey = null; },
    setPoints(xy, turnIds, memoryIds) {
      coordinates = xy; turns = turnIds; memories = memoryIds; lastKey = null;
      upload(turnBuffer, turns); upload(memoryBuffer, memories);
    },
    draw({ width, height, ratio, scale, pan, markerArea, rotation = IDENTITY_ROTATION, depth = null, autoOpacity = false, cloudOpacity = PLOT_STYLE.cloudOpacity, memoryOpacity = PLOT_STYLE.memoryOpacity }) {
      const key = [width, height, ratio, scale, ...pan, ...rotation, markerArea, autoOpacity, cloudOpacity, memoryOpacity, depth?.center, depth?.span, depth?.strength].join(",");
      if (lastKey === key || gl.isContextLost()) return;
      gl.viewport(0, 0, canvas.width, canvas.height); gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
      gl.uniform2f(uniform.viewport, width, height); gl.uniform2fv(uniform.pan, pan); gl.uniform1f(uniform.scale, scale);
      gl.uniformMatrix3fv(uniform.rotation, false, rotation);
      gl.uniform3f(uniform.depthCue, depth?.center || 0, depth?.span || 1, depth?.strength || 0);
      gl.uniform1f(uniform.shading, depth?.strength || 0);
      gl.uniform1f(uniform.adaptive, autoOpacity ? 1 : 0);
      for (const [buffer, ids, kind, opacity] of [[turnBuffer, turns, "turn", cloudOpacity], [memoryBuffer, memories, "memory", memoryOpacity]]) {
        if (!ids.length || opacity === 0) continue;
        const map = autoOpacity ? densityOpacityMap(coordinates, ids, { width, height, scale, pan, rotation }, opacity, kind) : null;
        summary[kind === "turn" ? "cloud" : "memory"] = map?.mean ?? opacity;
        if (map) {
          gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, densityTexture);
          gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, map.columns, map.rows, 0, gl.RGBA, gl.UNSIGNED_BYTE, map.pixels);
          gl.uniform2f(uniform.densityScale, 1 / (map.cell * ratio * map.columns), -1 / (map.cell * ratio * map.rows));
          gl.uniform2f(uniform.densityOffset, 1.5 / map.columns, (height / map.cell + 1.5) / map.rows);
        }
        gl.activeTexture(gl.TEXTURE0);
        const diameter = markerStyle(kind, markerArea).diameter;
        const glyph = texture(kind, diameter, ratio);
        gl.uniform1f(uniform.fillRatio, diameter * ratio / glyph.size);
        gl.bindTexture(gl.TEXTURE_2D, glyph.handle); gl.uniform1f(uniform.size, glyph.size); gl.uniform1f(uniform.opacity, opacity);
        gl.bindBuffer(gl.ARRAY_BUFFER, buffer); gl.vertexAttribPointer(position, 3, gl.FLOAT, false, 0, 0);
        gl.drawArrays(gl.POINTS, 0, ids.length);
      }
      lastKey = key;
    },
  };
  canvas.addEventListener("webglcontextlost", event => event.preventDefault());
  canvas.addEventListener("webglcontextrestored", () => {
    const renderer = createSpriteRenderer(canvas, gl); renderer.setPoints(coordinates, turns, memories);
    Object.assign(api, renderer); canvas.dispatchEvent(new Event("point-renderer-restored"));
  }, { once: true });
  return api;
}
