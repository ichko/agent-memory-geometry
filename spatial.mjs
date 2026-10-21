/* Orthographic 3D: genuine three-axis coordinates, with fixed-size scatter markers. */
export const IDENTITY_ROTATION = Object.freeze([1, 0, 0, 0, 1, 0, 0, 0, 1]);

export function orbitRotation(yaw, pitch) {
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  return [cy, sp * sy, -cp * sy, 0, cp, sp, sy, -sp * cy, cp * cy];
}

export function rotatedXY(point, rotation = IDENTITY_ROTATION) {
  const [x, y, z = 0] = point;
  return [rotation[0] * x + rotation[3] * y + rotation[6] * z, rotation[1] * x + rotation[4] * y + rotation[7] * z];
}

export function coordinateMoments(points) {
  const mean = [0, 0, 0], moments = [0, 0, 0, 0, 0, 0];
  if (!points.length) return { mean, covariance: moments };
  for (const [x, y, z] of points) {
    mean[0] += x; mean[1] += y; mean[2] += z;
    moments[0] += x * x; moments[1] += x * y; moments[2] += x * z;
    moments[3] += y * y; moments[4] += y * z; moments[5] += z * z;
  }
  for (let i = 0; i < 3; i++) mean[i] /= points.length;
  const products = [mean[0] ** 2, mean[0] * mean[1], mean[0] * mean[2], mean[1] ** 2, mean[1] * mean[2], mean[2] ** 2];
  return { mean, covariance: moments.map((value, i) => value / points.length - products[i]) };
}

export function depthCueForView(moments, rotation) {
  const [x, y, z] = [rotation[2], rotation[5], rotation[8]], c = moments.covariance;
  const variance = x * x * c[0] + y * y * c[3] + z * z * c[5] + 2 * (x * y * c[1] + x * z * c[2] + y * z * c[4]);
  return { center: x * moments.mean[0] + y * moments.mean[1] + z * moments.mean[2], span: Math.max(1e-6, 4 * Math.sqrt(Math.max(0, variance))), strength: 1 };
}

export function depthWeight(z, cue) {
  if (!cue) return 1;
  const t = Math.max(0, Math.min(1, .5 + (z - cue.center) / cue.span));
  return 1 + ((.55 + .9 * t ** 2) - 1) * cue.strength;
}

export function createSpatialPaths(canvas) {
  const gl = canvas.getContext("webgl", { alpha: true, premultipliedAlpha: false, antialias: true, depth: false });
  if (!gl) return null;
  const shader = (type, source) => {
    const handle = gl.createShader(type); gl.shaderSource(handle, source); gl.compileShader(handle);
    if (!gl.getShaderParameter(handle, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(handle));
    return handle;
  };
  const program = gl.createProgram();
  gl.attachShader(program, shader(gl.VERTEX_SHADER, `
    attribute vec3 position;
    uniform mat3 rotation;
    uniform vec2 viewport, pan;
    uniform float scale;
    uniform vec3 depthCue;
    varying mediump float depthWeight;
    void main() {
      vec3 rotated = rotation * position;
      vec2 pixel = rotated.xy * scale + pan;
      gl_Position = vec4(2.0 * pixel.x / viewport.x, -2.0 * pixel.y / viewport.y, 0.0, 1.0);
      float t = clamp(0.5 + (rotated.z - depthCue.x) / depthCue.y, 0.0, 1.0);
      depthWeight = mix(1.0, 0.65 + 0.7 * t, depthCue.z);
    }`));
  gl.attachShader(program, shader(gl.FRAGMENT_SHADER, `
    precision mediump float;
    uniform float opacity;
    varying mediump float depthWeight;
    void main() { gl_FragColor = vec4(vec3(127.0 / 255.0), opacity * depthWeight); }`));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
  gl.useProgram(program);
  const position = gl.getAttribLocation(program, "position"); gl.enableVertexAttribArray(position);
  const uniforms = Object.fromEntries(["rotation", "viewport", "pan", "scale", "opacity", "depthCue"].map(key => [key, gl.getUniformLocation(program, key)]));
  gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  const buffer = gl.createBuffer();
  let count = 0, lastKey = null, coordinates = [], sessions = [];
  const api = {
    invalidate() { lastKey = null; },
    setPaths(xyz, conversations) {
      coordinates = xyz; sessions = conversations;
      count = conversations.reduce((total, session) => total + Math.max(0, session.turns.length - 1) * 2, 0);
      const positions = new Float32Array(count * 3);
      let index = 0;
      for (const session of conversations) for (let i = 1; i < session.turns.length; i++) {
        for (const turn of [session.turns[i - 1], session.turns[i]]) {
          positions.set(xyz[turn.p], index); index += 3;
        }
      }
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer); gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW); lastKey = null;
    },
    draw({ width, height, ratio, scale, pan, rotation, pathAlpha, pathWidth = .25, depth = null }) {
      const key = [width, height, ratio, scale, ...pan, ...rotation, pathAlpha, pathWidth, depth?.center, depth?.span, depth?.strength].join(",");
      if (key === lastKey || gl.isContextLost()) return;
      gl.viewport(0, 0, canvas.width, canvas.height); gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
      gl.uniform2f(uniforms.viewport, width, height); gl.uniform2fv(uniforms.pan, pan);
      gl.uniform1f(uniforms.scale, scale); gl.uniformMatrix3fv(uniforms.rotation, false, rotation);
      gl.uniform3f(uniforms.depthCue, depth?.center || 0, depth?.span || 1, depth?.strength || 0);
      // WebGL's minimum line width is one device pixel. Match the lighter 0.25 CSS-pixel path coverage.
      gl.uniform1f(uniforms.opacity, pathAlpha * Math.min(1, pathWidth * ratio)); gl.lineWidth(1);
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer); gl.vertexAttribPointer(position, 3, gl.FLOAT, false, 0, 0);
      gl.drawArrays(gl.LINES, 0, count); lastKey = key;
    },
  };
  canvas.addEventListener("webglcontextlost", event => event.preventDefault());
  canvas.addEventListener("webglcontextrestored", () => {
    const renderer = createSpatialPaths(canvas);
    if (!renderer) return;
    renderer.setPaths(coordinates, sessions); Object.assign(api, renderer);
    canvas.dispatchEvent(new Event("point-renderer-restored"));
  }, { once: true });
  return api;
}
