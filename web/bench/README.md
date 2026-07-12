# Okada validation fixtures + Node checks

This folder holds the **CPU oracle and validation harness** behind the core
VisualQuakes bet: run the full Okada (1985) kernel once per pixel in a WebGL2
fragment shader, so dragging a slider is just a uniform update + one redraw.

The interactive, in-browser benchmark that used to live here
(`okada-bench.html`) grew up and moved to **[`../benchmark.html`](../benchmark.html)**
— it ships with the site, is linked from the app's About panel, and lets anyone
time the GPU shader against the JS float64 CPU port on their own machine (with
configurable grid sizes, timing frames, and CPU baseline), plus run the
GPU-vs-Python accuracy check.

## Why "nonlinear" doesn't hurt the GPU

GPUs care about *data parallelism* and *control-flow divergence*, not about
whether the math is linear. Okada is:

- **per-pixel independent** — each pixel depends only on its own `(e, n)` plus
  shared fault parameters (ideal data parallelism);
- **straight-line transcendental math** — `sqrt`, `atan`, `log`, `sin/cos`,
  divisions, which GPUs run in dedicated hardware units;
- **branch-free across pixels** — the only conditionals key on `dip` (a uniform),
  so every thread in a frame takes the same path. Zero divergence.

So "nonlinear" just means each thread does a fixed chunk of FLOPs — the GPU's
best case. The real questions are fp32 precision near the fault and whether
you're compute- vs overhead-bound, which `../benchmark.html` answers on your
hardware.

## Files

| File | Role |
|------|------|
| `../okada-shader.js` | GLSL ES 3.00 port of `displacement` / `tilt` / `strain` + LOS + colormaps (shared by the app and the benchmark page) |
| `okada85.mjs` | float64 JS port of `displacement` / `tilt` / `strain` (CPU baseline **and** correctness oracle) |
| `gen_reference.py` | writes `reference.json` from the Python float64 engine |
| `reference.json` | ground-truth fixture (params + displacement, LOS, tilt, strain on a grid) |
| `validate.mjs` | Node: JS port vs Python reference (displacement, tilt, strain) |
| `check-shader-algo.mjs` | Node: shader's restructured algebra vs reference (displacement, tilt, strain) |
| `test-usgs.mjs` | Node: USGS import logic (`../usgs.js` scaling relations, free-surface handling, ComCat parsing), network-free |

## Trust chain (the parts checkable without a GPU)

```bash
node web/bench/validate.mjs           # JS float64  vs Python float64
node web/bench/check-shader-algo.mjs  # shader algebra (float64) vs Python
```

Both pass to float64 round-off (~1e-19 km for displacement, ~4e-20 for tilt and
strain), so the shader's math is correct by construction. The **accuracy check**
on `../benchmark.html` adds the only device-dependent piece: fp32 vs float64 on
your GPU (the displacement/LOS path; tilt and strain share the same kernel
family and fp32 behavior). On real GPU hardware it reports LOS error of
**~0.003% of the signal peak** (sub-millimeter) — plain `highp` fp32 is
sufficient; no double-precision emulation needed.

## Run the browser benchmark locally

ES modules and `reference.json` need HTTP (not `file://`):

```bash
cd web
python -m http.server 8000
# open http://localhost:8000/benchmark.html
```

## Regenerate the fixture

If the scenario in `gen_reference.py` changes:

```bash
cd python && uv run python ../web/bench/gen_reference.py
```
