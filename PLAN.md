# VisualQuakes — Development Plan

A single-page, static website (GitHub Pages) that visualizes surface
deformation from one rectangular dislocation (Okada, 1985). The headline
interaction: **drag a slider, watch the InSAR fringes redraw in realtime.**

This document is the roadmap. Update it in the same commit whenever a step's
status changes. The rationale and measured numbers behind the rendering choice
live in [`benchmarks.md`](benchmarks.md).

---

## What's done

The interactive site is built and validated. In short:

- **Architecture chosen and proven.** The full Okada (1985) kernel runs once per
  pixel in a **WebGL2 fragment shader**; a slider drag is just a uniform update
  plus one redraw. Measured throughput leaves ~100–400× headroom over 60 fps at
  full-screen resolution, and fp32 error is ~0.003% of the signal peak — both
  confirmed on real GPU hardware. See [`benchmarks.md`](benchmarks.md).
- **Validated against the Python reference.** The GLSL kernel's algebra and the
  JS float64 port (`web/bench/okada85.mjs`, the CPU oracle/fallback) both match
  `python/src/visualquakes/okada85.py` to ~1e-19 km; on-GPU fp32 output is
  checked against the float64 oracle via the `web/bench/` harness.
- **The app** (`web/index.html` + `web/app.js`, zero-build ES modules sharing
  `web/okada-shader.js`): full-window canvas; continuous sliders with
  exact-entry number boxes for the fault + InSAR geometry; a Quantity toggle
  (displacement / tilt / strain) selecting wrapped-fringe / LOS / E / N / Up
  (displacement), East/North (tilt), or Eee/Enn/Ene/Areal (strain) views with
  adjustable saturation; semantic InSAR geometry (orbit pass, look direction,
  radar band, incidence, with an Advanced manual heading/λ override); pan,
  wheel-zoom, and **pinch-zoom**; fault-outline overlay with surface-trace
  singularity masking and a buried-fault constraint; colorbar legend, Mw
  readout, **scale bar**, presets, an About panel, and shareable URL state.
  The view auto-frames on the **fault trace** (top-edge midpoint) centered in the
  non-panel area of the window — a plain pan on load, and a zoom-to-fit when a
  scenario/USGS event is loaded so a large rupture fits with margin; **Reset
  view** restores all defaults and clears the URL. Mobile layout collapses to a
  single scrollable bottom-sheet panel with the legend + readout pinned at the
  top, above the sliders.
- **Deploy.** A GitHub Actions workflow (`.github/workflows/static.yml`)
  publishes `web/` as the site root on push to `main`.
- **Public benchmark page** (`web/benchmark.html`, linked from the About
  panel): a polished version of the original `web/bench/` proof-of-concept.
  Visitors time the GPU shader vs the JS float64 CPU port on their own machine
  (configurable grid sizes 256²–4096², timing frames, CPU-baseline cap with
  ∝-pixels extrapolation), see a log-scale GPU-vs-CPU comparison chart, stat
  tiles (speedup, Gpix/s, display-resolution headroom), a full results table,
  and can run the GPU-fp32-vs-Python-float64 accuracy check.

Conventions to preserve: render in **km** (fp32 mantissa), local Cartesian
`x=East, y=North, z=Up`, and reproduce `okada85.setup_args`'
centroid-to-Okada conversion exactly in any port.

---

## Next steps (Step 8)

Status keys: `[ ]` todo · `[~]` in progress · `[x]` done.

### Ship the live site `[x]`
- [x] Enable Pages (Settings → Pages → Source = GitHub Actions); live at
      https://ericlindsey.github.io/visualquakes/.
- [x] Add the live URL + a screenshot to the README.

### USGS event import `[x]`
- [x] Accept a **USGS event ID** (or a pasted event-page URL) and build an
      **approximate fault** from it (`web/usgs.js` + a Scenario-panel loader):
      fetches the ComCat detail feed (FDSN event service as fallback for alias
      IDs), reads the preferred moment-tensor / focal-mechanism product with an
      **NP1 / NP2 nodal-plane toggle**, and reports an inline error when the
      event has no mechanism. Length / width come from **Wells & Coppersmith
      (1994)** subsurface scaling for the mechanism type (classified from
      rake); if that width would breach the free surface at the catalog depth,
      width is trimmed and length extended (area preserved) to keep the rupture
      buried; slip is set from the scalar moment so the Mw readout matches the
      event wherever slider limits allow. Pure logic is unit-tested,
      network-free, by `node web/bench/test-usgs.mjs`.

### tilt / strain `[x]`
- [x] Port `okada85.tilt` and `okada85.strain` to JS (`web/bench/okada85.mjs`,
      validated vs the Python reference to ~4e-20 by `validate.mjs`) and to GLSL
      (`web/okada-shader.js`; the float64 mirror in `check-shader-algo.mjs`
      matches the reference, and the on-GPU fp32 output agrees to ~1e-6).
- [x] Surfaced as view modes: a Quantity toggle (Displacement / Tilt / Strain)
      drives the View segments — tilt shows East/North (uze, uzn); strain shows
      Eee, Enn, Ene (shear) and Areal (dilatation). Saturation switches units
      (cm / microradian / microstrain) and all of it round-trips through the URL.

### Other extensions `[ ]`
- [ ] GNSS-style displacement vector arrows overlaid on fringes.
- [ ] Multiple faults / simple finite-fault (sum of patches) — natural point to
      evaluate **WebGPU compute** if the patch count grows.
- [ ] Real basemap / coordinates (a true North arrow becomes meaningful once the
      view is geographically referenced); export PNG.
- [ ] Nice-to-have: keyboard nudges for fine slider control.

---

## Open questions

- **Units in the UI:** expose meters or km? (Internally render in km for fp32.)
- **Default InSAR geometry:** Sentinel-1 ascending (heading ≈ −12°, incidence
  ≈ 34°) is the current default.
- **Build tooling:** stay zero-build as long as possible; revisit if module
  organization or shader includes get unwieldy.

---

## Validation philosophy

The Python `visualquakes.okada85` (float64, 45 passing tests, traceable to the
original Beauducel/Lindsey Matlab reference) is the **source of truth**. Every
browser engine — JS, and especially the GLSL shader — is checked against it.
Correctness is pinned to the reference; performance is measured, not assumed.
