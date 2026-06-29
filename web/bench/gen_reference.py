"""Generate a ground-truth fixture from the float64 Okada reference.

Writes web/bench/reference.json: the fault/geometry parameters plus the Okada
surface displacement (ue, un, uz) and InSAR line-of-sight range change on a
coarse grid. The JS port and the GLSL shader are both validated against this
file, establishing the trust chain Python (float64) -> JS (float64) -> GLSL
(fp32).

All lengths are in kilometers; slip is in kilometers too (so 1e-3 == 1 m).
"""

import json
from pathlib import Path

import numpy as np

from visualquakes import okada85

# Canonical scenario, all lengths in km (keeps shader coordinates O(1-100)
# so fp32 retains mantissa bits). slip = 1e-3 km == 1 m.
PARAMS = {
    "depth": 5.0,
    "strike": 30.0,
    "dip": 45.0,
    "L": 10.0,
    "W": 6.0,
    "rake": 90.0,
    "slip": 1.0e-3,
    "open": 0.0,
    "nu": 0.25,
    # Map extent (km from centroid) and InSAR look geometry.
    "extent_km": 40.0,
    "insar_heading_deg": -12.0,   # Sentinel-1 ascending
    "insar_incidence_deg": 39.0,
    "wavelength_m": 0.0556,        # C-band; fringe = lambda/2 of range change
}

GRID_N = 64  # coarse validation grid


def los_unit_vector(heading_deg: float, incidence_deg: float) -> np.ndarray:
    """Return the unit look vector (East, North, Up) from ground to satellite.

    Args:
        heading_deg: Satellite heading (azimuth of flight) in degrees.
        incidence_deg: Radar incidence angle from vertical in degrees.

    Returns:
        Length-3 array [le, ln, lu] of the line-of-sight unit vector.
    """
    heading = np.radians(heading_deg)
    inc = np.radians(incidence_deg)
    # Look direction points to the right of the flight track (right-looking).
    az = heading + np.pi / 2.0
    le = -np.sin(inc) * np.sin(az)
    ln = -np.sin(inc) * np.cos(az)
    lu = np.cos(inc)
    return np.array([le, ln, lu])


def main() -> None:
    p = PARAMS
    coords = np.linspace(-p["extent_km"], p["extent_km"], GRID_N)
    e, n = np.meshgrid(coords, coords)
    args = (
        e, n, p["depth"], p["strike"], p["dip"], p["L"], p["W"],
        p["rake"], p["slip"], p["open"], p["nu"],
    )
    ue, un, uz = okada85.displacement(*args)
    uze, uzn = okada85.tilt(*args)
    unn, une, uen, uee = okada85.strain(*args)
    look = los_unit_vector(p["insar_heading_deg"], p["insar_incidence_deg"])
    los = ue * look[0] + un * look[1] + uz * look[2]

    flat = lambda a: a.ravel(order="C").tolist()
    out = {
        "params": p,
        "look_enu": look.tolist(),
        "grid_n": GRID_N,
        "coords_km": coords.tolist(),
        "ue": flat(ue),
        "un": flat(un),
        "uz": flat(uz),
        "los": flat(los),
        "uze": flat(uze),
        "uzn": flat(uzn),
        "unn": flat(unn),
        "une": flat(une),
        "uen": flat(uen),
        "uee": flat(uee),
    }
    dest = Path(__file__).parent / "reference.json"
    dest.write_text(json.dumps(out))
    print(f"wrote {dest} ({dest.stat().st_size} bytes), grid {GRID_N}x{GRID_N}")
    print(f"los range change: min={los.min():.3e} max={los.max():.3e} km")
    print(f"tilt max |uze|={np.abs(uze).max():.3e} |uzn|={np.abs(uzn).max():.3e} rad")
    print(f"strain max |uee|={np.abs(uee).max():.3e} |unn|={np.abs(unn).max():.3e}")


if __name__ == "__main__":
    main()
