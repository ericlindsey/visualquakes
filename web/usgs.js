// USGS ComCat event import: fetch an event by ID, read its focal mechanism
// (moment-tensor or focal-mechanism product), and derive plausible fault
// dimensions/slip from the magnitude via Wells & Coppersmith (1994) scaling,
// respecting the free surface. Pure derivation logic is kept separate from the
// fetch so it can be unit-tested in node (web/bench/test-usgs.mjs).

const DEG = Math.PI / 180;
const MU = 3e10; // shear modulus, Pa (matches app.js momentMagnitude)
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

// Wells & Coppersmith (1994), BSSA 84(4), Table 2A: log10(dim km) = a + b*Mw.
// RLD = subsurface rupture length, RW = downdip rupture width — the right pair
// for a buried rectangular dislocation (surface rupture length would be
// shorter and only defined for surface-breaking events).
const WC94 = {
  ss:      { L: [-2.57, 0.62], W: [-0.76, 0.27] }, // strike-slip
  reverse: { L: [-2.42, 0.58], W: [-1.61, 0.41] },
  normal:  { L: [-1.88, 0.50], W: [-1.14, 0.35] },
};

// Classify by rake: within 45 deg of pure thrust (+90) => reverse, of pure
// normal (-90) => normal, else treat as strike-slip (includes oblique).
export function classifyMechanism(rake) {
  if (rake >= 45 && rake <= 135) return "reverse";
  if (rake <= -45 && rake >= -135) return "normal";
  return "ss";
}

// Subsurface rupture length/width (km) for a moment magnitude + mechanism.
export function scaledDimensions(mw, mech) {
  const c = WC94[mech] ?? WC94.ss;
  return {
    length: Math.pow(10, c.L[0] + c.L[1] * mw),
    width: Math.pow(10, c.W[0] + c.W[1] * mw),
  };
}

export const moment = (mw) => Math.pow(10, 1.5 * mw + 9.1); // N*m

// Build fault-slider parameters from one nodal plane + magnitude + depth.
//
// Free-surface handling: the fault centroid stays at the catalog depth. If the
// Wells & Coppersmith width would poke the top edge above the surface
// (depth - sin(dip)*W/2 < margin), the width is reduced to fit and the length
// increased to preserve the rupture area, so the aspect ratio absorbs the
// constraint instead of the depth. Slip is then set from the seismic moment
// and the final area, so the Mw readout matches the event wherever the slider
// limits allow.
//
// `limits` maps field name -> [min, max] (from app.js FAULT_FIELDS); `margin`
// is the app's SURFACE_MARGIN burial margin in km.
// Returns { fault, notes } where notes lists any compromises made.
export function faultFromMechanism(plane, mw, depthKm, limits, margin = 0.05) {
  const notes = [];
  const strike = clamp(plane.strike, ...limits.strike);
  const dip = clamp(plane.dip, ...limits.dip);
  const rake = clamp(plane.rake, ...limits.rake);

  const depth = clamp(depthKm, ...limits.depth);
  if (depthKm > limits.depth[1]) {
    notes.push(`depth ${Math.round(depthKm)} km capped at ${limits.depth[1]} km`);
  }

  const mech = classifyMechanism(rake);
  let { length, width } = scaledDimensions(mw, mech);
  const area = length * width;

  // Widest fault that keeps the top edge buried at this centroid depth/dip.
  const maxWidth = (2 * (depth - margin)) / Math.max(Math.sin(dip * DEG), 1e-6);
  if (width > maxWidth) {
    width = Math.max(maxWidth, limits.width[0]);
    length = area / width; // preserve area => same slip for the same moment
    notes.push("width trimmed to keep the rupture below the surface");
  }

  const finalL = clamp(length, ...limits.length);
  const finalW = clamp(width, ...limits.width);
  if (finalL < length || finalW < width) {
    notes.push("size capped at slider limits");
  }

  const slip = moment(mw) / (MU * finalL * 1e3 * finalW * 1e3);
  const finalSlip = clamp(slip, ...limits.slip);
  if (finalSlip < slip) {
    notes.push(`slip capped at ${limits.slip[1]} m — Mw readout will run low`);
  }

  return {
    fault: { strike, dip, rake, depth, length: finalL, width: finalW, slip: finalSlip, open: 0 },
    notes,
  };
}

// Accept a bare event ID ("ci38457511") or a pasted event-page URL
// ("https://earthquake.usgs.gov/earthquakes/eventpage/ci38457511/executive").
export function extractEventId(text) {
  const t = text.trim();
  const m = t.match(/eventpage\/([^/?#\s]+)/);
  return m ? m[1] : t.replace(/[/?#\s].*$/, "");
}

// --------------------------------------------------- moment tensor -> planes
// Some ComCat moment-tensor products (regional networks especially) carry only
// the tensor components, not pre-derived nodal planes. Recover the two nodal
// planes from the tensor so those events still load.

// Cyclic Jacobi eigen-decomposition of a symmetric 3x3 matrix. Returns
// eigenvalues and their eigenvectors (vecs[j] is the unit vector for values[j]).
function jacobiEigen(A) {
  const a = A.map((r) => r.slice());
  const v = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  for (let sweep = 0; sweep < 100; sweep++) {
    const off = Math.abs(a[0][1]) + Math.abs(a[0][2]) + Math.abs(a[1][2]);
    if (off < 1e-20) break;
    for (const [p, q] of [[0, 1], [0, 2], [1, 2]]) {
      if (Math.abs(a[p][q]) < 1e-300) continue;
      const phi = 0.5 * Math.atan2(2 * a[p][q], a[q][q] - a[p][p]);
      const c = Math.cos(phi), s = Math.sin(phi);
      for (let k = 0; k < 3; k++) {
        const akp = a[k][p], akq = a[k][q];
        a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq;
      }
      for (let k = 0; k < 3; k++) {
        const apk = a[p][k], aqk = a[q][k];
        a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk;
      }
      for (let k = 0; k < 3; k++) {
        const vkp = v[k][p], vkq = v[k][q];
        v[k][p] = c * vkp - s * vkq; v[k][q] = s * vkp + c * vkq;
      }
    }
  }
  return {
    values: [a[0][0], a[1][1], a[2][2]],
    vecs: [0, 1, 2].map((j) => [v[0][j], v[1][j], v[2][j]]),
  };
}

const DEG_PER_RAD = 180 / Math.PI;
const unit = (u) => { const m = Math.hypot(...u) || 1; return u.map((x) => x / m); };

// One nodal plane (strike/dip/rake, degrees) from an upward fault normal and
// slip vector, both in (North, East, Down) -- Aki & Richards convention.
function vecToSDR(n, s) {
  if (n[2] > 0) { n = n.map((x) => -x); s = s.map((x) => -x); } // normal points up
  const [n1, n2] = n;
  const sinDip = Math.hypot(n1, n2);
  const dip = Math.atan2(sinDip, -n[2]);        // 0..90
  const strike = Math.atan2(-n1, n2);
  const cp = Math.cos(strike), sp = Math.sin(strike);
  // sin(rake) = -slip_down / sin(dip); a near-horizontal fault (sinDip -> 0)
  // has an undefined strike/rake, so fall back to the raw down component.
  const sinLambda = sinDip > 1e-9 ? -s[2] / sinDip : -s[2];
  const rake = Math.atan2(sinLambda, s[0] * cp + s[1] * sp);
  return {
    strike: ((strike * DEG_PER_RAD) % 360 + 360) % 360,
    dip: dip * DEG_PER_RAD,
    rake: rake * DEG_PER_RAD,
  };
}

// The two nodal planes of a moment tensor given in the GCMT/USGS spherical
// system (r, t, p) = (up, south, east). Exported for the node test.
export function nodalPlanesFromTensor(t) {
  const M = [
    [t.mrr, t.mrt, t.mrp],
    [t.mrt, t.mtt, t.mtp],
    [t.mrp, t.mtp, t.mpp],
  ];
  const { values, vecs } = jacobiEigen(M);
  const order = [0, 1, 2].sort((i, j) => values[i] - values[j]);
  const toNED = (u) => [-u[1], u[2], -u[0]]; // (r,t,p) -> (N,E,D)
  const P = unit(toNED(vecs[order[0]])); // pressure axis (min eigenvalue)
  const T = unit(toNED(vecs[order[2]])); // tension axis (max eigenvalue)
  const normal = unit(T.map((x, i) => x + P[i]));
  const slip = unit(T.map((x, i) => x - P[i]));
  return [vecToSDR(normal, slip), vecToSDR(slip, normal)];
}

// Pull the two nodal planes out of one product's properties: the pre-derived
// nodal-plane-* fields if present (older focal mechanisms name the rake
// "slip"), otherwise computed from the tensor-* components. Returns null when
// the product carries neither.
function planesFromProps(mp) {
  const num = (k) => {
    const v = parseFloat(mp[k]);
    return Number.isFinite(v) ? v : null;
  };
  const planes = [1, 2].map((n) => ({
    strike: num(`nodal-plane-${n}-strike`),
    dip: num(`nodal-plane-${n}-dip`),
    rake: num(`nodal-plane-${n}-rake`) ?? num(`nodal-plane-${n}-slip`),
  }));
  if (planes.every((p) => p.strike != null && p.dip != null && p.rake != null)) {
    return planes;
  }
  const comps = ["mrr", "mtt", "mpp", "mrt", "mrp", "mtp"].map((c) => num(`tensor-${c}`));
  if (comps.every((v) => v != null)) {
    const [mrr, mtt, mpp, mrt, mrp, mtp] = comps;
    return nodalPlanesFromTensor({ mrr, mtt, mpp, mrt, mrp, mtp });
  }
  return null;
}

// Parse a ComCat detail GeoJSON feature into { title, mw, depthKm, planes }.
// Scans every moment-tensor product (preferred) then every focal-mechanism
// product for the first one with usable nodal planes -- a regional network's
// tensor-only product and a global product can coexist on one event. Product
// property values are strings in ComCat. Throws a user-facing message when no
// usable mechanism exists. Exported for the node test.
export function parseEventFeature(geo) {
  const props = geo?.properties;
  if (!props) throw new Error("Unexpected response from the USGS API.");
  const products = props.products ?? {};
  const title = props.title ?? "USGS event";
  const candidates = [...(products["moment-tensor"] ?? []),
                      ...(products["focal-mechanism"] ?? [])];
  if (candidates.length === 0) {
    throw new Error(`No focal mechanism is available for “${title}”.`);
  }

  let chosen = null, planes = null;
  for (const prod of candidates) {
    planes = planesFromProps(prod.properties ?? {});
    if (planes) { chosen = prod; break; }
  }
  if (!planes) {
    throw new Error(`The mechanism for “${title}” has no usable nodal planes.`);
  }

  const num = (k) => {
    const v = parseFloat((chosen.properties ?? {})[k]);
    return Number.isFinite(v) ? v : null;
  };
  const mw = num("derived-magnitude") ?? (Number.isFinite(props.mag) ? props.mag : null);
  const depthKm = num("derived-depth") ??
    (Number.isFinite(geo.geometry?.coordinates?.[2]) ? geo.geometry.coordinates[2] : null);
  if (mw == null || depthKm == null) {
    throw new Error(`“${title}” is missing a magnitude or depth.`);
  }
  return { title, mw, depthKm, planes };
}

const DETAIL_URL = (id) =>
  `https://earthquake.usgs.gov/earthquakes/feed/v1.0/detail/${encodeURIComponent(id)}.geojson`;

// Fetch an event by ComCat ID from the realtime detail feed. This is the only
// endpoint that carries the moment-tensor / focal-mechanism products (the FDSN
// summary GeoJSON does not), and it resolves an event's associated/alias IDs
// server-side, so one request suffices. The feed is CORS-enabled
// (Access-Control-Allow-Origin: *), so it works from a static page.
//
// A hard timeout via AbortController turns a stalled request into a visible
// error instead of leaving the UI stuck on "Looking up event...". A rejected
// fetch is a network failure or a CORS block, not a bad ID -- say so plainly.
export async function fetchUsgsEvent(id, timeoutMs = 12000) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  let resp;
  try {
    resp = await fetch(DETAIL_URL(id), { signal: ctrl.signal });
  } catch (err) {
    throw new Error(err?.name === "AbortError"
      ? "The USGS request timed out — check your connection and try again."
      : "Could not reach the USGS API (network error or blocked by the browser).");
  } finally {
    clearTimeout(timer);
  }
  if (resp.status === 404 || resp.status === 400) {
    throw new Error(`Event “${id}” was not found — check the ID.`);
  }
  if (!resp.ok) throw new Error(`USGS API error (HTTP ${resp.status}).`);
  return parseEventFeature(await resp.json());
}
