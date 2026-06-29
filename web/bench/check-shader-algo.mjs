// Verify the *algorithm* used by okada-shader.js (FRAG_SRC) is mathematically
// correct, independent of GPU/fp32. This mirrors the shader's restructured
// kernel exactly -- combined subAt corner evaluation, i4 threaded into I3, i5
// into I1, the once-per-pixel atan term, and the q / xi guards -- in float64,
// then compares to the Python reference fixture. A pass means the shader's
// algebra matches the reference; the remaining fp32 gap is what the in-browser
// "Validate" button measures.  Run: node web/bench/check-shader-algo.mjs

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const DEG = Math.PI / 180;

// Globals set per-fault, mirroring the shader uniforms.
let SD, CD, NU, VERTICAL;

function I5f(xi, eta, q, R, db) {
  const X = Math.sqrt(xi * xi + q * q);
  const xs = xi === 0 ? 1e-12 : xi;
  if (!VERTICAL)
    return (1 - 2 * NU) * 2 / CD *
      Math.atan((eta * (X + q * CD) + X * (R + X) * SD) / (xs * (R + X) * CD));
  return -(1 - 2 * NU) * xi * SD / (R + db);
}
function I4f(db, eta, q, R) {
  if (!VERTICAL)
    return (1 - 2 * NU) / CD * (Math.log(R + db) - SD * Math.log(R + eta));
  return -(1 - 2 * NU) * q / (R + db);
}
function I3f(eta, q, R, i4) {
  const yb = eta * CD + q * SD, db = eta * SD - q * CD;
  if (!VERTICAL)
    return (1 - 2 * NU) * (yb / (CD * (R + db)) - Math.log(R + eta)) + SD / CD * i4;
  return (1 - 2 * NU) / 2 *
    (eta / (R + db) + yb * q / ((R + db) * (R + db)) - Math.log(R + eta));
}
function I1f(xi, eta, q, R, i5) {
  const db = eta * SD - q * CD;
  if (!VERTICAL)
    return (1 - 2 * NU) * (-xi / (CD * (R + db))) - SD / CD * i5;
  return -(1 - 2 * NU) / 2 * xi * q / ((R + db) * (R + db));
}

// Returns {ss:[x,y,z], ds:[...], tf:[...]} at one Chinnery corner.
function subAt(xi, eta, q) {
  const R = Math.sqrt(xi * xi + eta * eta + q * q);
  const db = eta * SD - q * CD;
  const Re = R + eta, Rx = R + xi;
  const at = Math.atan(xi * eta / (q * R));
  const i5 = I5f(xi, eta, q, R, db);
  const i4 = I4f(db, eta, q, R);
  const i3 = I3f(eta, q, R, i4);
  const i2 = (1 - 2 * NU) * (-Math.log(Re)) - i3;
  const i1 = I1f(xi, eta, q, R, i5);
  return {
    ss: [xi * q / (R * Re) + at + i1 * SD,
         (eta * CD + q * SD) * q / (R * Re) + q * CD / Re + i2 * SD,
         db * q / (R * Re) + q * SD / Re + i4 * SD],
    ds: [q / R - i3 * SD * CD,
         (eta * CD + q * SD) * q / (R * Rx) + CD * at - i1 * SD * CD,
         db * q / (R * Rx) + SD * at - i5 * SD * CD],
    tf: [q * q / (R * Re) - i3 * SD * SD,
         -db * q / (R * Rx) - SD * (xi * q / (R * Re) - at) - i1 * SD * SD,
         (eta * CD + q * SD) * q / (R * Rx) + CD * (xi * q / (R * Re) - at) - i5 * SD * SD],
  };
}

// --- Tilt + strain subfunctions, mirroring the shader's Af/Kf/Jf helpers ---
function Af(x, R) { const Rx = R + x; return (2 * R + x) / (R * R * R * Rx * Rx); }
function K1f(xi, eta, q, R) {
  const db = eta * SD - q * CD;
  if (!VERTICAL)
    return (1 - 2 * NU) * xi / CD * (1 / (R * (R + db)) - SD / (R * (R + eta)));
  return (1 - 2 * NU) * xi * q / ((R + db) * (R + db));
}
function K3f(xi, eta, q, R) {
  const db = eta * SD - q * CD, yb = eta * CD + q * SD;
  if (!VERTICAL)
    return (1 - 2 * NU) / CD * (q / (R * (R + eta)) - yb / (R * (R + db)));
  return (1 - 2 * NU) * SD / (R + db) * (xi * xi / (R * (R + db)) - 1);
}
function K2f(eta, q, R, k3) {
  return (1 - 2 * NU) * (-SD / R + q * CD / (R * (R + eta))) - k3;
}
function J1f(xi, eta, q, R, k3) {
  const db = eta * SD - q * CD;
  if (!VERTICAL)
    return (1 - 2 * NU) / CD * (xi * xi / (R * (R + db) * (R + db)) - 1 / (R + db)) -
      SD / CD * k3;
  return (1 - 2 * NU) / 2 * q / ((R + db) * (R + db)) * (2 * xi * xi / (R * (R + db)) - 1);
}
function J2f(xi, eta, q, R, k1) {
  const db = eta * SD - q * CD, yb = eta * CD + q * SD;
  if (!VERTICAL)
    return (1 - 2 * NU) / CD * xi * yb / (R * (R + db) * (R + db)) - SD / CD * k1;
  return (1 - 2 * NU) / 2 * xi * SD / ((R + db) * (R + db)) * (2 * q * q / (R * (R + db)) - 1);
}
function J3f(xi, eta, q, R, j2) { return (1 - 2 * NU) * -xi / (R * (R + eta)) - j2; }
function J4f(eta, q, R, j1) {
  return (1 - 2 * NU) * (-CD / R - q * SD / (R * (R + eta))) - j1;
}

function tiltAt(xi, eta, q) {
  const R = Math.sqrt(xi * xi + eta * eta + q * q), R3 = R * R * R;
  const db = eta * SD - q * CD, yb = eta * CD + q * SD;
  const Re = R + eta, Rx = R + xi, ae = Af(eta, R), ax = Af(xi, R);
  const k1 = K1f(xi, eta, q, R), k3 = K3f(xi, eta, q, R), k2 = K2f(eta, q, R, k3);
  return {
    ss: [-xi * q * q * ae * CD + (xi * q / R3 - k1) * SD,
         db * q / R3 * CD + (xi * xi * q * ae * CD - SD / R + yb * q / R3 - k2) * SD],
    ds: [db * q / R3 + q * SD / (R * Re) + k3 * SD * CD,
         yb * db * q * ax - (2 * db / (R * Rx) + xi * SD / (R * Re)) * SD + k1 * SD * CD],
    tf: [q * q / R3 * SD - q * q * q * ae * CD + k3 * SD * SD,
         (yb * SD + db * CD) * q * q * ax + xi * q * q * ae * SD * CD -
           (2 * q / (R * Rx) - k1) * SD * SD],
  };
}
function strainAt(xi, eta, q) {
  const R = Math.sqrt(xi * xi + eta * eta + q * q), R3 = R * R * R;
  const db = eta * SD - q * CD, yb = eta * CD + q * SD;
  const Re = R + eta, Rx = R + xi, ae = Af(eta, R), ax = Af(xi, R);
  const k1 = K1f(xi, eta, q, R), k3 = K3f(xi, eta, q, R);
  const j1 = J1f(xi, eta, q, R, k3), j2 = J2f(xi, eta, q, R, k1);
  const j3 = J3f(xi, eta, q, R, j2), j4 = J4f(eta, q, R, j1);
  return {
    ss: [xi * xi * q * ae - j1 * SD,
         xi * xi * xi * db / (R3 * (eta * eta + q * q)) - (xi * xi * xi * ae + j2) * SD,
         xi * q / R3 * CD + (xi * q * q * ae - j2) * SD,
         yb * q / R3 * CD + (q * q * q * ae * SD - 2 * q * SD / (R * Re) -
           (xi * xi + eta * eta) / R3 * CD - j4) * SD],
    ds: [xi * q / R3 + j3 * SD * CD,
         yb * q / R3 - SD / R + j1 * SD * CD,
         yb * q / R3 + q * CD / (R * Re) + j1 * SD * CD,
         yb * yb * q * ax - (2 * yb / (R * Rx) + xi * CD / (R * Re)) * SD + j2 * SD * CD],
    tf: [xi * q * q * ae + j3 * SD * SD,
         -db * q / R3 - xi * xi * q * ae * SD + j1 * SD * SD,
         q * q / R3 * CD + q * q * q * ae * SD + j1 * SD * SD,
         (yb * CD - db * SD) * q * q * ax - q * (2 * SD * CD) / (R * Rx) -
           (xi * q * q * ae - j2) * SD * SD],
  };
}

// Common setup mirroring okadaXPQ + the per-fault uniforms; returns {x,p,q,...}.
function setup(e, n, p) {
  const strike = p.strike * DEG, dip = p.dip * DEG, rake = p.rake * DEG;
  const sstr = Math.sin(strike), cstr = Math.cos(strike);
  SD = Math.sin(dip); CD = Math.cos(dip); NU = p.nu;
  VERTICAL = !(Math.cos(dip) > Number.EPSILON);
  const U1s = Math.cos(rake) * p.slip / (2 * Math.PI);
  const U2s = Math.sin(rake) * p.slip / (2 * Math.PI);
  const U3s = p.open / (2 * Math.PI);
  const d = p.depth + SD * p.W / 2;
  const ec = e + cstr * CD * p.W / 2, nc = n - sstr * CD * p.W / 2;
  const x = cstr * nc + sstr * ec + p.L / 2;
  const y = sstr * nc - cstr * ec + CD * p.W;
  const pp = y * CD + d * SD;
  let q = y * SD - d * CD;
  if (Math.abs(q) < 1e-12) q = 1e-12;
  return { x, p: pp, q, sstr, cstr, U1s, U2s, U3s };
}

function shaderDisp(e, n, p) {
  const g = setup(e, n, p);
  const a = subAt(g.x, g.p, g.q), b = subAt(g.x, g.p - p.W, g.q);
  const c = subAt(g.x - p.L, g.p, g.q), dd = subAt(g.x - p.L, g.p - p.W, g.q);
  const comb = (k, i) => a[k][i] - b[k][i] - c[k][i] + dd[k][i];
  const ux = -g.U1s * comb("ss", 0) - g.U2s * comb("ds", 0) + g.U3s * comb("tf", 0);
  const uy = -g.U1s * comb("ss", 1) - g.U2s * comb("ds", 1) + g.U3s * comb("tf", 1);
  const uz = -g.U1s * comb("ss", 2) - g.U2s * comb("ds", 2) + g.U3s * comb("tf", 2);
  return [g.sstr * ux - g.cstr * uy, g.cstr * ux + g.sstr * uy, uz];
}

function shaderTilt(e, n, p) {
  const g = setup(e, n, p);
  const a = tiltAt(g.x, g.p, g.q), b = tiltAt(g.x, g.p - p.W, g.q);
  const c = tiltAt(g.x - p.L, g.p, g.q), dd = tiltAt(g.x - p.L, g.p - p.W, g.q);
  const comb = (k, i) => a[k][i] - b[k][i] - c[k][i] + dd[k][i];
  const uzx = -g.U1s * comb("ss", 0) - g.U2s * comb("ds", 0) + g.U3s * comb("tf", 0);
  const uzy = -g.U1s * comb("ss", 1) - g.U2s * comb("ds", 1) + g.U3s * comb("tf", 1);
  return [-g.sstr * uzx + g.cstr * uzy, -g.cstr * uzx - g.sstr * uzy];
}

function shaderStrain(e, n, p) {
  const g = setup(e, n, p);
  const a = strainAt(g.x, g.p, g.q), b = strainAt(g.x, g.p - p.W, g.q);
  const c = strainAt(g.x - p.L, g.p, g.q), dd = strainAt(g.x - p.L, g.p - p.W, g.q);
  const comb = (i) => -g.U1s * (a.ss[i] - b.ss[i] - c.ss[i] + dd.ss[i])
    - g.U2s * (a.ds[i] - b.ds[i] - c.ds[i] + dd.ds[i])
    + g.U3s * (a.tf[i] - b.tf[i] - c.tf[i] + dd.tf[i]);
  const uxx = comb(0), uxy = comb(1), uyx = comb(2), uyy = comb(3);
  const s2 = 2 * g.sstr * g.cstr, c2 = g.cstr * g.cstr, ss2 = g.sstr * g.sstr;
  return [
    c2 * uxx + s2 * (uxy + uyx) / 2 + ss2 * uyy,            // unn
    ss2 * uyx + s2 * (uxx - uyy) / 2 - c2 * uxy,            // une
    -c2 * uyx + s2 * (uxx - uyy) / 2 + ss2 * uxy,           // uen
    ss2 * uxx - s2 * (uyx + uxy) / 2 + c2 * uyy,            // uee
  ];
}

const here = dirname(fileURLToPath(import.meta.url));
const ref = JSON.parse(readFileSync(join(here, "reference.json"), "utf8"));
const p = ref.params, N = ref.grid_n, coords = ref.coords_km;
let maxAbs = 0, tiltMax = 0, strainMax = 0;
for (let j = 0; j < N; j++)
  for (let i = 0; i < N; i++) {
    const idx = j * N + i;
    const [ue, un, uz] = shaderDisp(coords[i], coords[j], p);
    maxAbs = Math.max(maxAbs, Math.abs(ue - ref.ue[idx]),
      Math.abs(un - ref.un[idx]), Math.abs(uz - ref.uz[idx]));
    const [uze, uzn] = shaderTilt(coords[i], coords[j], p);
    tiltMax = Math.max(tiltMax, Math.abs(uze - ref.uze[idx]), Math.abs(uzn - ref.uzn[idx]));
    const [unn, une, uen, uee] = shaderStrain(coords[i], coords[j], p);
    strainMax = Math.max(strainMax, Math.abs(unn - ref.unn[idx]),
      Math.abs(une - ref.une[idx]), Math.abs(uen - ref.uen[idx]), Math.abs(uee - ref.uee[idx]));
  }
console.log(`shader algorithm (float64) vs reference, ${N}x${N}`);
console.log(`max abs displacement error: ${maxAbs.toExponential(3)} km`);
console.log(`max abs tilt error: ${tiltMax.toExponential(3)} rad`);
console.log(`max abs strain error: ${strainMax.toExponential(3)}`);
const ok = maxAbs < 1e-15 && tiltMax < 1e-15 && strainMax < 1e-15;
console.log(ok ? "PASS: shader algebra matches reference"
               : "FAIL: shader algebra diverges -- transcription bug");
process.exit(ok ? 0 : 1);
