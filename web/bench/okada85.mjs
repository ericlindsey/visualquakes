// Float64 JavaScript port of visualquakes.okada85.displacement (Okada, 1985).
//
// A faithful, scalar transcription of python/src/visualquakes/okada85.py used
// (a) as the CPU baseline in the benchmark and (b) as the correctness oracle
// for the GLSL shader. Kept structurally identical to the Python reference:
// setup_args -> chinnery -> strike-slip / dip-slip / tensile subfunctions.
//
// Conventions match the reference: e, n are East/North offsets from the fault
// centroid; depth is centroid depth (positive down); strike/dip/rake in
// degrees; all lengths in consistent units.

const EPS = Number.EPSILON; // == numpy np.spacing(1)
const DEG = Math.PI / 180;

// I... displacement subfunctions [Okada eqs (28)-(29)].
function I5(xi, eta, q, dip, nu, R, db) {
  const sd = Math.sin(dip), cd = Math.cos(dip);
  const X = Math.sqrt(xi * xi + q * q);
  const xs = xi === 0 ? xi + EPS : xi; // guard the cos(dip)>0 branch
  if (cd > EPS) {
    return (1 - 2 * nu) * 2 / cd *
      Math.atan((eta * (X + q * cd) + X * (R + X) * sd) /
                (xs * (R + X) * cd));
  }
  return -(1 - 2 * nu) * xi * sd / (R + db);
}

function I4(db, eta, q, dip, nu, R) {
  const sd = Math.sin(dip), cd = Math.cos(dip);
  if (cd > EPS) {
    return (1 - 2 * nu) / cd * (Math.log(R + db) - sd * Math.log(R + eta));
  }
  return -(1 - 2 * nu) * q / (R + db);
}

function I3(eta, q, dip, nu, R) {
  const sd = Math.sin(dip), cd = Math.cos(dip);
  const yb = eta * cd + q * sd;
  const db = eta * sd - q * cd;
  if (cd > EPS) {
    return (1 - 2 * nu) * (yb / (cd * (R + db)) - Math.log(R + eta)) +
      sd / cd * I4(db, eta, q, dip, nu, R);
  }
  return (1 - 2 * nu) / 2 *
    (eta / (R + db) + yb * q / ((R + db) * (R + db)) - Math.log(R + eta));
}

function I2(eta, q, dip, nu, R) {
  return (1 - 2 * nu) * (-Math.log(R + eta)) - I3(eta, q, dip, nu, R);
}

function I1(xi, eta, q, dip, nu, R) {
  const sd = Math.sin(dip), cd = Math.cos(dip);
  const db = eta * sd - q * cd;
  if (cd > EPS) {
    return (1 - 2 * nu) * (-xi / (cd * (R + db))) -
      sd / cd * I5(xi, eta, q, dip, nu, R, db);
  }
  return -(1 - 2 * nu) / 2 * xi * q / ((R + db) * (R + db));
}

// Displacement subfunctions [Okada eqs (25)-(27)].
function ux_ss(xi, eta, q, dip, nu) {
  const sd = Math.sin(dip);
  const R = Math.sqrt(xi * xi + eta * eta + q * q);
  return xi * q / (R * (R + eta)) + Math.atan(xi * eta / (q * R)) +
    I1(xi, eta, q, dip, nu, R) * sd;
}
function uy_ss(xi, eta, q, dip, nu) {
  const sd = Math.sin(dip), cd = Math.cos(dip);
  const R = Math.sqrt(xi * xi + eta * eta + q * q);
  return (eta * cd + q * sd) * q / (R * (R + eta)) + q * cd / (R + eta) +
    I2(eta, q, dip, nu, R) * sd;
}
function uz_ss(xi, eta, q, dip, nu) {
  const sd = Math.sin(dip), cd = Math.cos(dip);
  const R = Math.sqrt(xi * xi + eta * eta + q * q);
  const db = eta * sd - q * cd;
  return (eta * sd - q * cd) * q / (R * (R + eta)) + q * sd / (R + eta) +
    I4(db, eta, q, dip, nu, R) * sd;
}
function ux_ds(xi, eta, q, dip, nu) {
  const sd = Math.sin(dip), cd = Math.cos(dip);
  const R = Math.sqrt(xi * xi + eta * eta + q * q);
  return q / R - I3(eta, q, dip, nu, R) * sd * cd;
}
function uy_ds(xi, eta, q, dip, nu) {
  const sd = Math.sin(dip), cd = Math.cos(dip);
  const R = Math.sqrt(xi * xi + eta * eta + q * q);
  return (eta * cd + q * sd) * q / (R * (R + xi)) +
    cd * Math.atan(xi * eta / (q * R)) -
    I1(xi, eta, q, dip, nu, R) * sd * cd;
}
function uz_ds(xi, eta, q, dip, nu) {
  const sd = Math.sin(dip), cd = Math.cos(dip);
  const R = Math.sqrt(xi * xi + eta * eta + q * q);
  const db = eta * sd - q * cd;
  return db * q / (R * (R + xi)) + sd * Math.atan(xi * eta / (q * R)) -
    I5(xi, eta, q, dip, nu, R, db) * sd * cd;
}
function ux_tf(xi, eta, q, dip, nu) {
  const sd = Math.sin(dip);
  const R = Math.sqrt(xi * xi + eta * eta + q * q);
  return q * q / (R * (R + eta)) - I3(eta, q, dip, nu, R) * sd * sd;
}
function uy_tf(xi, eta, q, dip, nu) {
  const sd = Math.sin(dip), cd = Math.cos(dip);
  const R = Math.sqrt(xi * xi + eta * eta + q * q);
  return -(eta * sd - q * cd) * q / (R * (R + xi)) -
    sd * (xi * q / (R * (R + eta)) - Math.atan(xi * eta / (q * R))) -
    I1(xi, eta, q, dip, nu, R) * sd * sd;
}
function uz_tf(xi, eta, q, dip, nu) {
  const sd = Math.sin(dip), cd = Math.cos(dip);
  const R = Math.sqrt(xi * xi + eta * eta + q * q);
  const db = eta * sd - q * cd;
  return (eta * cd + q * sd) * q / (R * (R + xi)) +
    cd * (xi * q / (R * (R + eta)) - Math.atan(xi * eta / (q * R))) -
    I5(xi, eta, q, dip, nu, R, db) * sd * sd;
}

// Chinnery's notation [Okada eq (24)]: f(x,p) - f(x,p-W) - f(x-L,p) + f(x-L,p-W).
function chinnery(f, x, p, L, W, q, dip, nu) {
  return f(x, p, q, dip, nu) - f(x, p - W, q, dip, nu) -
    f(x - L, p, q, dip, nu) + f(x - L, p - W, q, dip, nu);
}

/**
 * Okada (1985) surface displacement for a single rectangular dislocation.
 * Scalar; mirrors okada85.displacement in the Python reference.
 * @returns {[number, number, number]} [ue, un, uz]
 */
export function displacement(e, n, depth, strike, dip, L, W, rake, slip, open, nu = 0.25) {
  strike *= DEG;
  dip *= DEG;
  rake *= DEG;
  const sstr = Math.sin(strike), cstr = Math.cos(strike);
  const sdip = Math.sin(dip), cdip = Math.cos(dip);

  const U1s = Math.cos(rake) * slip / (2 * Math.PI);
  const U2s = Math.sin(rake) * slip / (2 * Math.PI);
  const U3s = open / (2 * Math.PI);

  // Centroid -> Okada reference point (center of top edge), then x, p, q.
  const d = depth + sdip * W / 2;
  const ec = e + cstr * cdip * W / 2;
  const nc = n - sstr * cdip * W / 2;
  const x = cstr * nc + sstr * ec + L / 2;
  const y = sstr * nc - cstr * ec + cdip * W;
  const p = y * cdip + d * sdip;
  const q = y * sdip - d * cdip;

  const ux = -U1s * chinnery(ux_ss, x, p, L, W, q, dip, nu)
             - U2s * chinnery(ux_ds, x, p, L, W, q, dip, nu)
             + U3s * chinnery(ux_tf, x, p, L, W, q, dip, nu);
  const uy = -U1s * chinnery(uy_ss, x, p, L, W, q, dip, nu)
             - U2s * chinnery(uy_ds, x, p, L, W, q, dip, nu)
             + U3s * chinnery(uy_tf, x, p, L, W, q, dip, nu);
  const uz = -U1s * chinnery(uz_ss, x, p, L, W, q, dip, nu)
             - U2s * chinnery(uz_ds, x, p, L, W, q, dip, nu)
             + U3s * chinnery(uz_tf, x, p, L, W, q, dip, nu);

  const ue = sstr * ux - cstr * uy;
  const un = cstr * ux + sstr * uy;
  return [ue, un, uz];
}

// =====================================================================
// Tilt and strain. Same structure as displacement: setup_args -> chinnery
// over the tilt/strain subfunctions, then rotate Okada (x, y) axes to (E, N).
// Mirrors okada85.tilt / okada85.strain in the Python reference.

// A and the K... tilt subfunctions [Okada eqs (40)-(41)].
function A(x, R) {
  const Rx = R + x;
  return (2 * R + x) / (R * R * R * Rx * Rx);
}
function K1(xi, eta, q, dip, nu, R) {
  const sd = Math.sin(dip), cd = Math.cos(dip);
  const db = eta * sd - q * cd;
  if (cd > EPS) {
    return (1 - 2 * nu) * xi / cd * (1 / (R * (R + db)) - sd / (R * (R + eta)));
  }
  return (1 - 2 * nu) * xi * q / ((R + db) * (R + db));
}
function K3(xi, eta, q, dip, nu, R) {
  const sd = Math.sin(dip), cd = Math.cos(dip);
  const db = eta * sd - q * cd;
  const yb = eta * cd + q * sd;
  if (cd > EPS) {
    return (1 - 2 * nu) / cd * (q / (R * (R + eta)) - yb / (R * (R + db)));
  }
  return (1 - 2 * nu) * sd / (R + db) * (xi * xi / (R * (R + db)) - 1);
}
function K2(xi, eta, q, dip, nu, R) {
  const sd = Math.sin(dip), cd = Math.cos(dip);
  return (1 - 2 * nu) * (-sd / R + q * cd / (R * (R + eta))) -
    K3(xi, eta, q, dip, nu, R);
}

// Tilt subfunctions [Okada eqs (37)-(39)].
function uzx_ss(xi, eta, q, dip, nu) {
  const sd = Math.sin(dip), cd = Math.cos(dip);
  const R = Math.sqrt(xi * xi + eta * eta + q * q);
  return -xi * q * q * A(eta, R) * cd +
    (xi * q / (R * R * R) - K1(xi, eta, q, dip, nu, R)) * sd;
}
function uzy_ss(xi, eta, q, dip, nu) {
  const sd = Math.sin(dip), cd = Math.cos(dip);
  const R = Math.sqrt(xi * xi + eta * eta + q * q);
  const db = eta * sd - q * cd, yb = eta * cd + q * sd;
  return db * q / (R * R * R) * cd +
    (xi * xi * q * A(eta, R) * cd - sd / R + yb * q / (R * R * R) -
      K2(xi, eta, q, dip, nu, R)) * sd;
}
function uzx_ds(xi, eta, q, dip, nu) {
  const sd = Math.sin(dip), cd = Math.cos(dip);
  const R = Math.sqrt(xi * xi + eta * eta + q * q);
  const db = eta * sd - q * cd;
  return db * q / (R * R * R) + q * sd / (R * (R + eta)) +
    K3(xi, eta, q, dip, nu, R) * sd * cd;
}
function uzy_ds(xi, eta, q, dip, nu) {
  const sd = Math.sin(dip), cd = Math.cos(dip);
  const R = Math.sqrt(xi * xi + eta * eta + q * q);
  const db = eta * sd - q * cd, yb = eta * cd + q * sd;
  return yb * db * q * A(xi, R) -
    (2 * db / (R * (R + xi)) + xi * sd / (R * (R + eta))) * sd +
    K1(xi, eta, q, dip, nu, R) * sd * cd;
}
function uzx_tf(xi, eta, q, dip, nu) {
  const sd = Math.sin(dip), cd = Math.cos(dip);
  const R = Math.sqrt(xi * xi + eta * eta + q * q);
  return q * q / (R * R * R) * sd - q * q * q * A(eta, R) * cd +
    K3(xi, eta, q, dip, nu, R) * sd * sd;
}
function uzy_tf(xi, eta, q, dip, nu) {
  const sd = Math.sin(dip), cd = Math.cos(dip);
  const R = Math.sqrt(xi * xi + eta * eta + q * q);
  const db = eta * sd - q * cd, yb = eta * cd + q * sd;
  return (yb * sd + db * cd) * q * q * A(xi, R) +
    xi * q * q * A(eta, R) * sd * cd -
    (2 * q / (R * (R + xi)) - K1(xi, eta, q, dip, nu, R)) * sd * sd;
}

/**
 * Okada (1985) surface tilt (d uz / dx) for a single rectangular dislocation.
 * @returns {[number, number]} [uze, uzn]
 */
export function tilt(e, n, depth, strike, dip, L, W, rake, slip, open, nu = 0.25) {
  strike *= DEG;
  dip *= DEG;
  rake *= DEG;
  const sstr = Math.sin(strike), cstr = Math.cos(strike);
  const sdip = Math.sin(dip), cdip = Math.cos(dip);

  const U1s = Math.cos(rake) * slip / (2 * Math.PI);
  const U2s = Math.sin(rake) * slip / (2 * Math.PI);
  const U3s = open / (2 * Math.PI);

  const d = depth + sdip * W / 2;
  const ec = e + cstr * cdip * W / 2;
  const nc = n - sstr * cdip * W / 2;
  const x = cstr * nc + sstr * ec + L / 2;
  const y = sstr * nc - cstr * ec + cdip * W;
  const p = y * cdip + d * sdip;
  const q = y * sdip - d * cdip;

  const uzx = -U1s * chinnery(uzx_ss, x, p, L, W, q, dip, nu)
              - U2s * chinnery(uzx_ds, x, p, L, W, q, dip, nu)
              + U3s * chinnery(uzx_tf, x, p, L, W, q, dip, nu);
  const uzy = -U1s * chinnery(uzy_ss, x, p, L, W, q, dip, nu)
              - U2s * chinnery(uzy_ds, x, p, L, W, q, dip, nu)
              + U3s * chinnery(uzy_tf, x, p, L, W, q, dip, nu);

  const uze = -sstr * uzx + cstr * uzy;
  const uzn = -cstr * uzx - sstr * uzy;
  return [uze, uzn];
}

// J... strain subfunctions [Okada eqs (34)-(35)].
function J1(xi, eta, q, dip, nu, R) {
  const sd = Math.sin(dip), cd = Math.cos(dip);
  const db = eta * sd - q * cd;
  if (cd > EPS) {
    return (1 - 2 * nu) / cd * (xi * xi / (R * (R + db) * (R + db)) - 1 / (R + db)) -
      sd / cd * K3(xi, eta, q, dip, nu, R);
  }
  return (1 - 2 * nu) / 2 * q / ((R + db) * (R + db)) *
    (2 * xi * xi / (R * (R + db)) - 1);
}
function J2(xi, eta, q, dip, nu, R) {
  const sd = Math.sin(dip), cd = Math.cos(dip);
  const db = eta * sd - q * cd, yb = eta * cd + q * sd;
  if (cd > EPS) {
    return (1 - 2 * nu) / cd * xi * yb / (R * (R + db) * (R + db)) -
      sd / cd * K1(xi, eta, q, dip, nu, R);
  }
  return (1 - 2 * nu) / 2 * xi * sd / ((R + db) * (R + db)) *
    (2 * q * q / (R * (R + db)) - 1);
}
function J3(xi, eta, q, dip, nu, R) {
  return (1 - 2 * nu) * -xi / (R * (R + eta)) - J2(xi, eta, q, dip, nu, R);
}
function J4(xi, eta, q, dip, nu, R) {
  const sd = Math.sin(dip), cd = Math.cos(dip);
  return (1 - 2 * nu) * (-cd / R - q * sd / (R * (R + eta))) -
    J1(xi, eta, q, dip, nu, R);
}

// Strain subfunctions [Okada eqs (31)-(33)].
function uxx_ss(xi, eta, q, dip, nu) {
  const sd = Math.sin(dip);
  const R = Math.sqrt(xi * xi + eta * eta + q * q);
  return xi * xi * q * A(eta, R) - J1(xi, eta, q, dip, nu, R) * sd;
}
function uxy_ss(xi, eta, q, dip, nu) {
  const sd = Math.sin(dip), cd = Math.cos(dip);
  const R = Math.sqrt(xi * xi + eta * eta + q * q);
  const db = eta * sd - q * cd;
  return xi * xi * xi * db / (R * R * R * (eta * eta + q * q)) -
    (xi * xi * xi * A(eta, R) + J2(xi, eta, q, dip, nu, R)) * sd;
}
function uyx_ss(xi, eta, q, dip, nu) {
  const sd = Math.sin(dip), cd = Math.cos(dip);
  const R = Math.sqrt(xi * xi + eta * eta + q * q);
  return xi * q / (R * R * R) * cd +
    (xi * q * q * A(eta, R) - J2(xi, eta, q, dip, nu, R)) * sd;
}
function uyy_ss(xi, eta, q, dip, nu) {
  const sd = Math.sin(dip), cd = Math.cos(dip);
  const R = Math.sqrt(xi * xi + eta * eta + q * q);
  const yb = eta * cd + q * sd;
  return yb * q / (R * R * R) * cd +
    (q * q * q * A(eta, R) * sd - 2 * q * sd / (R * (R + eta)) -
      (xi * xi + eta * eta) / (R * R * R) * cd - J4(xi, eta, q, dip, nu, R)) * sd;
}
function uxx_ds(xi, eta, q, dip, nu) {
  const sd = Math.sin(dip), cd = Math.cos(dip);
  const R = Math.sqrt(xi * xi + eta * eta + q * q);
  return xi * q / (R * R * R) + J3(xi, eta, q, dip, nu, R) * sd * cd;
}
function uxy_ds(xi, eta, q, dip, nu) {
  const sd = Math.sin(dip), cd = Math.cos(dip);
  const R = Math.sqrt(xi * xi + eta * eta + q * q);
  const yb = eta * cd + q * sd;
  return yb * q / (R * R * R) - sd / R + J1(xi, eta, q, dip, nu, R) * sd * cd;
}
function uyx_ds(xi, eta, q, dip, nu) {
  const sd = Math.sin(dip), cd = Math.cos(dip);
  const R = Math.sqrt(xi * xi + eta * eta + q * q);
  const yb = eta * cd + q * sd;
  return yb * q / (R * R * R) + q * cd / (R * (R + eta)) +
    J1(xi, eta, q, dip, nu, R) * sd * cd;
}
function uyy_ds(xi, eta, q, dip, nu) {
  const sd = Math.sin(dip), cd = Math.cos(dip);
  const R = Math.sqrt(xi * xi + eta * eta + q * q);
  const yb = eta * cd + q * sd;
  return yb * yb * q * A(xi, R) -
    (2 * yb / (R * (R + xi)) + xi * cd / (R * (R + eta))) * sd +
    J2(xi, eta, q, dip, nu, R) * sd * cd;
}
function uxx_tf(xi, eta, q, dip, nu) {
  const sd = Math.sin(dip);
  const R = Math.sqrt(xi * xi + eta * eta + q * q);
  return xi * q * q * A(eta, R) + J3(xi, eta, q, dip, nu, R) * sd * sd;
}
function uxy_tf(xi, eta, q, dip, nu) {
  const sd = Math.sin(dip), cd = Math.cos(dip);
  const R = Math.sqrt(xi * xi + eta * eta + q * q);
  const db = eta * sd - q * cd;
  return -db * q / (R * R * R) - xi * xi * q * A(eta, R) * sd +
    J1(xi, eta, q, dip, nu, R) * sd * sd;
}
function uyx_tf(xi, eta, q, dip, nu) {
  const sd = Math.sin(dip), cd = Math.cos(dip);
  const R = Math.sqrt(xi * xi + eta * eta + q * q);
  return q * q / (R * R * R) * cd + q * q * q * A(eta, R) * sd +
    J1(xi, eta, q, dip, nu, R) * sd * sd;
}
function uyy_tf(xi, eta, q, dip, nu) {
  const sd = Math.sin(dip), cd = Math.cos(dip);
  const R = Math.sqrt(xi * xi + eta * eta + q * q);
  const db = eta * sd - q * cd, yb = eta * cd + q * sd;
  return (yb * cd - db * sd) * q * q * A(xi, R) -
    q * (2 * sd * cd) / (R * (R + xi)) -
    (xi * q * q * A(eta, R) - J2(xi, eta, q, dip, nu, R)) * sd * sd;
}

/**
 * Okada (1985) surface strain for a single rectangular dislocation.
 * Sign convention matches the Python reference: positive = compression.
 * @returns {[number, number, number, number]} [unn, une, uen, uee]
 */
export function strain(e, n, depth, strike, dip, L, W, rake, slip, open, nu = 0.25) {
  strike *= DEG;
  dip *= DEG;
  rake *= DEG;
  const sstr = Math.sin(strike), cstr = Math.cos(strike);
  const sdip = Math.sin(dip), cdip = Math.cos(dip);

  const U1s = Math.cos(rake) * slip / (2 * Math.PI);
  const U2s = Math.sin(rake) * slip / (2 * Math.PI);
  const U3s = open / (2 * Math.PI);

  const d = depth + sdip * W / 2;
  const ec = e + cstr * cdip * W / 2;
  const nc = n - sstr * cdip * W / 2;
  const x = cstr * nc + sstr * ec + L / 2;
  const y = sstr * nc - cstr * ec + cdip * W;
  const p = y * cdip + d * sdip;
  const q = y * sdip - d * cdip;

  const uxx = -U1s * chinnery(uxx_ss, x, p, L, W, q, dip, nu)
              - U2s * chinnery(uxx_ds, x, p, L, W, q, dip, nu)
              + U3s * chinnery(uxx_tf, x, p, L, W, q, dip, nu);
  const uxy = -U1s * chinnery(uxy_ss, x, p, L, W, q, dip, nu)
              - U2s * chinnery(uxy_ds, x, p, L, W, q, dip, nu)
              + U3s * chinnery(uxy_tf, x, p, L, W, q, dip, nu);
  const uyx = -U1s * chinnery(uyx_ss, x, p, L, W, q, dip, nu)
              - U2s * chinnery(uyx_ds, x, p, L, W, q, dip, nu)
              + U3s * chinnery(uyx_tf, x, p, L, W, q, dip, nu);
  const uyy = -U1s * chinnery(uyy_ss, x, p, L, W, q, dip, nu)
              - U2s * chinnery(uyy_ds, x, p, L, W, q, dip, nu)
              + U3s * chinnery(uyy_tf, x, p, L, W, q, dip, nu);

  const s2 = 2 * sstr * cstr;       // sin(2*strike)
  const c2 = cstr * cstr, ss2 = sstr * sstr;
  const unn = c2 * uxx + s2 * (uxy + uyx) / 2 + ss2 * uyy;
  const une = ss2 * uyx + s2 * (uxx - uyy) / 2 - c2 * uxy;
  const uen = -c2 * uyx + s2 * (uxx - uyy) / 2 + ss2 * uxy;
  const uee = ss2 * uxx - s2 * (uyx + uxy) / 2 + c2 * uyy;
  return [unn, une, uen, uee];
}
