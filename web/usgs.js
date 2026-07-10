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

// Parse a ComCat detail GeoJSON feature into { title, mw, depthKm, planes }.
// Product property values are strings in ComCat. Older focal-mechanism
// products call the rake "slip". Throws with a user-facing message when no
// usable mechanism exists. Exported for the node test.
export function parseEventFeature(geo) {
  const props = geo?.properties;
  if (!props) throw new Error("Unexpected response from the USGS API.");
  const products = props.products ?? {};
  // Products are ordered by preference; moment tensors over first-motion
  // focal mechanisms.
  const src = (products["moment-tensor"] ?? products["focal-mechanism"] ?? [])[0];
  const title = props.title ?? "USGS event";
  if (!src) throw new Error(`No focal mechanism is available for “${title}”.`);

  const mp = src.properties ?? {};
  const num = (k) => {
    const v = parseFloat(mp[k]);
    return Number.isFinite(v) ? v : null;
  };
  const planes = [1, 2].map((n) => ({
    strike: num(`nodal-plane-${n}-strike`),
    dip: num(`nodal-plane-${n}-dip`),
    rake: num(`nodal-plane-${n}-rake`) ?? num(`nodal-plane-${n}-slip`),
  }));
  if (planes.some((p) => p.strike == null || p.dip == null || p.rake == null)) {
    throw new Error(`The mechanism for “${title}” has no usable nodal planes.`);
  }

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
const FDSN_URL = (id) =>
  `https://earthquake.usgs.gov/fdsnws/event/1/query?eventid=${encodeURIComponent(id)}&format=geojson`;

// Fetch an event by ID. Tries the realtime detail feed first, then the FDSN
// event service (which also resolves alias/merged event IDs).
export async function fetchUsgsEvent(id) {
  let lastStatus = 0;
  for (const url of [DETAIL_URL(id), FDSN_URL(id)]) {
    let resp;
    try {
      resp = await fetch(url);
    } catch {
      throw new Error("Could not reach the USGS API — check your connection.");
    }
    if (resp.ok) return parseEventFeature(await resp.json());
    lastStatus = resp.status;
  }
  if (lastStatus === 404 || lastStatus === 400) {
    throw new Error(`Event “${id}” was not found — check the ID.`);
  }
  throw new Error(`USGS API error (HTTP ${lastStatus}).`);
}
