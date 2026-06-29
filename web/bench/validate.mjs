// Validate the JS float64 port against the Python float64 reference fixture.
//   node web/bench/validate.mjs
// Regenerate reference.json first via gen_reference.py if params change.

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { displacement, tilt, strain } from "./okada85.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const ref = JSON.parse(readFileSync(join(here, "reference.json"), "utf8"));
const p = ref.params;
const coords = ref.coords_km;
const N = ref.grid_n;
const look = ref.look_enu;

let maxAbs = 0, maxRel = 0;
let losMaxAbs = 0;
let tiltMaxAbs = 0, strainMaxAbs = 0;
const refLosMax = Math.max(...ref.los.map(Math.abs));

for (let j = 0; j < N; j++) {
  for (let i = 0; i < N; i++) {
    const idx = j * N + i;
    const args = [coords[i], coords[j], p.depth, p.strike, p.dip, p.L, p.W,
                  p.rake, p.slip, p.open, p.nu];
    const [ue, un, uz] = displacement(...args);
    for (const [got, exp] of [[ue, ref.ue[idx]], [un, ref.un[idx]], [uz, ref.uz[idx]]]) {
      const abs = Math.abs(got - exp);
      maxAbs = Math.max(maxAbs, abs);
      if (Math.abs(exp) > 1e-12) maxRel = Math.max(maxRel, abs / Math.abs(exp));
    }
    const los = ue * look[0] + un * look[1] + uz * look[2];
    losMaxAbs = Math.max(losMaxAbs, Math.abs(los - ref.los[idx]));

    const [uze, uzn] = tilt(...args);
    tiltMaxAbs = Math.max(tiltMaxAbs,
      Math.abs(uze - ref.uze[idx]), Math.abs(uzn - ref.uzn[idx]));
    const [unn, une, uen, uee] = strain(...args);
    strainMaxAbs = Math.max(strainMaxAbs,
      Math.abs(unn - ref.unn[idx]), Math.abs(une - ref.une[idx]),
      Math.abs(uen - ref.uen[idx]), Math.abs(uee - ref.uee[idx]));
  }
}

console.log(`grid ${N}x${N}, ${N * N} points`);
console.log(`displacement max abs error: ${maxAbs.toExponential(3)} km`);
console.log(`displacement max rel error: ${maxRel.toExponential(3)}`);
console.log(`LOS max abs error: ${losMaxAbs.toExponential(3)} km` +
            ` (signal peak ${refLosMax.toExponential(3)} km)`);
console.log(`tilt max abs error: ${tiltMaxAbs.toExponential(3)} rad`);
console.log(`strain max abs error: ${strainMaxAbs.toExponential(3)}`);

// float64 vs float64: absolute agreement to round-off. Relative error is not
// gated because it explodes harmlessly where the true field is ~0. Tilt/strain
// carry one extra derivative so their round-off floor sits a little higher.
const ok = maxAbs < 1e-15 && tiltMaxAbs < 1e-15 && strainMaxAbs < 1e-15;
console.log(ok ? "PASS: JS port matches Python reference" :
                 "FAIL: JS port diverges from Python reference");
process.exit(ok ? 0 : 1);
