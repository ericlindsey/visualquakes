// Unit tests for the USGS import logic in web/usgs.js (scaling relations,
// free-surface handling, ComCat parsing). Network-free.
//   node web/bench/test-usgs.mjs

import {
  classifyMechanism, scaledDimensions, moment, faultFromMechanism,
  extractEventId, parseEventFeature,
} from "../usgs.js";

// Mirror app.js FAULT_FIELDS ranges.
const LIMITS = {
  strike: [0, 360], dip: [1, 90], rake: [-180, 180], depth: [0.2, 40],
  length: [0.5, 80], width: [0.5, 50], slip: [0, 15],
};
const MARGIN = 0.05;

let failures = 0;
function check(name, cond, detail = "") {
  if (!cond) { failures++; console.error(`FAIL  ${name}  ${detail}`); }
  else console.log(`ok    ${name}`);
}
const close = (a, b, tol) => Math.abs(a - b) <= tol;

// Mw implied by a fault, matching app.js momentMagnitude (mu = 30 GPa).
function impliedMw(f) {
  const Mo = 3e10 * f.slip * (f.length * 1000) * (f.width * 1000);
  return (2 / 3) * (Math.log10(Mo) - 9.1);
}

// ---- mechanism classification --------------------------------------------
check("rake 90 -> reverse", classifyMechanism(90) === "reverse");
check("rake -90 -> normal", classifyMechanism(-90) === "normal");
check("rake 0 -> ss", classifyMechanism(0) === "ss");
check("rake 180 -> ss", classifyMechanism(180) === "ss");
check("rake -170 -> ss", classifyMechanism(-170) === "ss");
check("rake 60 -> reverse", classifyMechanism(60) === "reverse");

// ---- Wells & Coppersmith values (Table 2A, subsurface RLD / RW) ----------
// Mw 7 strike-slip: L = 10^(-2.57+0.62*7) ~ 58.9 km, W = 10^(-0.76+0.27*7) ~ 13.5 km
{
  const { length, width } = scaledDimensions(7, "ss");
  check("WC94 M7 ss length", close(length, 58.88, 0.1), `${length}`);
  check("WC94 M7 ss width", close(width, 13.49, 0.1), `${width}`);
}
// Mw 6.5 reverse: L = 10^(-2.42+0.58*6.5) ~ 22.4 km, W = 10^(-1.61+0.41*6.5) ~ 11.4 km
{
  const { length, width } = scaledDimensions(6.5, "reverse");
  check("WC94 M6.5 rev length", close(length, 22.39, 0.1), `${length}`);
  check("WC94 M6.5 rev width", close(width, 11.43, 0.1), `${width}`);
}

// ---- moment ----------------------------------------------------------------
check("Mo(Mw 7)", close(moment(7), 3.981e19, 2e16), `${moment(7)}`);

// ---- fault derivation: deep event, no clamping -----------------------------
{
  const { fault, notes } = faultFromMechanism(
    { strike: 320, dip: 81, rake: -173 }, 7.1, 12, LIMITS, MARGIN);
  check("deep ss: no notes", notes.length === 0, notes.join("; "));
  check("deep ss: plane copied",
    fault.strike === 320 && fault.dip === 81 && fault.rake === -173);
  check("deep ss: depth kept", fault.depth === 12);
  check("deep ss: Mw matches", close(impliedMw(fault), 7.1, 1e-9),
    `${impliedMw(fault)}`);
  const top = fault.depth - Math.sin(fault.dip * Math.PI / 180) * fault.width / 2;
  check("deep ss: buried", top >= MARGIN - 1e-9, `top edge ${top}`);
}

// ---- fault derivation: shallow thrust hits the free surface ----------------
{
  const mw = 6.8, depth = 4, dip = 60;
  const { fault, notes } = faultFromMechanism(
    { strike: 0, dip, rake: 90 }, mw, depth, LIMITS, MARGIN);
  const unconstrained = scaledDimensions(mw, "reverse");
  check("shallow thrust: width would breach unconstrained",
    depth - Math.sin(dip * Math.PI / 180) * unconstrained.width / 2 < MARGIN);
  const top = fault.depth - Math.sin(dip * Math.PI / 180) * fault.width / 2;
  check("shallow thrust: top edge at margin", close(top, MARGIN, 1e-9), `${top}`);
  check("shallow thrust: area preserved", close(
    fault.length * fault.width,
    unconstrained.length * unconstrained.width, 1e-6));
  check("shallow thrust: longer than W&C", fault.length > unconstrained.length);
  check("shallow thrust: Mw matches", close(impliedMw(fault), mw, 1e-9),
    `${impliedMw(fault)}`);
  check("shallow thrust: noted", notes.some((n) => n.includes("width")),
    notes.join("; "));
}

// ---- fault derivation: great earthquake saturates the sliders --------------
{
  const { fault, notes } = faultFromMechanism(
    { strike: 0, dip: 15, rake: 90 }, 9.0, 25, LIMITS, MARGIN);
  check("M9: length at max", fault.length === LIMITS.length[1], `${fault.length}`);
  check("M9: width at max", fault.width === LIMITS.width[1], `${fault.width}`);
  check("M9: slip at max", fault.slip === LIMITS.slip[1], `${fault.slip}`);
  check("M9: notes mention caps",
    notes.some((n) => n.includes("size")) && notes.some((n) => n.includes("slip")),
    notes.join("; "));
}

// ---- fault derivation: deep event depth capped ------------------------------
{
  const { fault, notes } = faultFromMechanism(
    { strike: 10, dip: 45, rake: -90 }, 7.0, 600, LIMITS, MARGIN);
  check("deep: depth capped", fault.depth === LIMITS.depth[1], `${fault.depth}`);
  check("deep: noted", notes.some((n) => n.includes("depth")), notes.join("; "));
}

// ---- event ID extraction ----------------------------------------------------
check("id passthrough", extractEventId(" ci38457511 ") === "ci38457511");
check("event-page URL", extractEventId(
  "https://earthquake.usgs.gov/earthquakes/eventpage/us7000abcd/moment-tensor")
  === "us7000abcd");

// ---- ComCat parsing ---------------------------------------------------------
// ComCat product property values are strings.
const feature = (products, mag = 7.1, depth = 8) => ({
  properties: { title: `M ${mag} - test region`, mag, products },
  geometry: { coordinates: [-117.6, 35.77, depth] },
});
const mtProps = {
  "derived-magnitude": "7.05", "derived-depth": "9.5", "scalar-moment": "4.4e19",
  "nodal-plane-1-strike": "322", "nodal-plane-1-dip": "81", "nodal-plane-1-rake": "-173",
  "nodal-plane-2-strike": "231", "nodal-plane-2-dip": "83", "nodal-plane-2-rake": "-9",
};
{
  const ev = parseEventFeature(feature({ "moment-tensor": [{ properties: mtProps }] }));
  check("parse: mt magnitude preferred", ev.mw === 7.05, `${ev.mw}`);
  check("parse: mt depth preferred", ev.depthKm === 9.5, `${ev.depthKm}`);
  check("parse: NP1", ev.planes[0].strike === 322 && ev.planes[0].dip === 81
    && ev.planes[0].rake === -173);
  check("parse: NP2", ev.planes[1].strike === 231 && ev.planes[1].rake === -9);
}
{
  // focal-mechanism fallback with the legacy "slip" key and no derived fields.
  const fm = {
    "nodal-plane-1-strike": "10", "nodal-plane-1-dip": "40", "nodal-plane-1-slip": "95",
    "nodal-plane-2-strike": "184", "nodal-plane-2-dip": "50", "nodal-plane-2-slip": "86",
  };
  const ev = parseEventFeature(feature({ "focal-mechanism": [{ properties: fm }] }));
  check("parse: fm rake from slip key", ev.planes[0].rake === 95);
  check("parse: mw falls back to event mag", ev.mw === 7.1);
  check("parse: depth falls back to hypocenter", ev.depthKm === 8);
}
{
  let msg = "";
  try { parseEventFeature(feature({ origin: [{ properties: {} }] })); }
  catch (e) { msg = e.message; }
  check("parse: no mechanism throws", msg.includes("No focal mechanism"), msg);
}

if (failures) { console.error(`\n${failures} failure(s)`); process.exit(1); }
console.log("\nall usgs import checks passed");
