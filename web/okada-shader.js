// GLSL ES 3.00 (WebGL2) port of okada85.displacement, plus InSAR LOS, fringe /
// amplitude colormaps, and a fault-outline overlay. fp32 throughout (browser
// shaders have no f64). Branches key only on `u_vertical`, a uniform derived
// from dip, so control flow never diverges across pixels -- the GPU-friendly
// property that makes the nonlinear Okada kernel parallelize cleanly.
//
// Shared by the app (web/index.html) and the benchmark (web/bench/). The
// numeric core is validated against the Python reference; the `u_mode==1` path
// returns raw displacement before any display code runs.

export const VERT_SRC = `#version 300 es
// Fullscreen triangle from gl_VertexID; no attribute buffers needed.
void main() {
  vec2 p = vec2((gl_VertexID == 1) ? 3.0 : -1.0,
                (gl_VertexID == 2) ? 3.0 : -1.0);
  gl_Position = vec4(p, 0.0, 1.0);
}`;

export const FRAG_SRC = `#version 300 es
precision highp float;
precision highp int;

uniform vec2  u_resolution;  // drawing buffer size (px)
uniform vec2  u_extent;      // half-extent of the view in km (ex, ey)
uniform vec2  u_origin;      // view center offset in km (pan)

uniform float u_depth, u_L, u_W, u_nu;
uniform float u_sstr, u_cstr, u_sdip, u_cdip;  // sin/cos of strike, dip
uniform float u_U1s, u_U2s, u_U3s;             // pre-scaled slip components
uniform int   u_vertical;                      // 1 if cos(dip) ~ 0
uniform vec3  u_look;                           // InSAR LOS unit vector (E,N,U)
uniform float u_fringe;                         // range change per fringe (km)
uniform int   u_mode;                           // 0 = display, 1 = raw floats
uniform int   u_quantity;   // 0 displacement, 1 tilt, 2 strain

// Display controls (ignored when u_mode == 1). u_view indexes within the
// quantity: displacement = {fringes,LOS,E,N,Up}; tilt = {E,N}; strain =
// {Eee,Enn,Ene,areal}.
uniform int   u_view;
uniform float u_ampScale;   // full-scale (km) for the diverging colormap
uniform int   u_showFault;  // 1 to draw the fault-outline overlay
uniform vec2  u_c0, u_c1, u_c2, u_c3; // surface-projected fault corners (km)
uniform float u_kmPerPx;    // view scale, for constant-width overlay lines

out vec4 outColor;

const float XI_EPS = 1.0e-12; // turns exact xi==0 into a tiny nonzero value

// --- I... subfunctions [Okada eqs (28)-(29)]; branch on uniform u_vertical ---
float I5f(float xi, float eta, float q, float R, float db) {
  float X = sqrt(xi * xi + q * q);
  float xs = (xi == 0.0) ? XI_EPS : xi;
  if (u_vertical == 0) {
    return (1.0 - 2.0 * u_nu) * 2.0 / u_cdip *
      atan((eta * (X + q * u_cdip) + X * (R + X) * u_sdip) /
           (xs * (R + X) * u_cdip));
  }
  return -(1.0 - 2.0 * u_nu) * xi * u_sdip / (R + db);
}
float I4f(float db, float eta, float q, float R) {
  if (u_vertical == 0) {
    return (1.0 - 2.0 * u_nu) / u_cdip *
      (log(R + db) - u_sdip * log(R + eta));
  }
  return -(1.0 - 2.0 * u_nu) * q / (R + db);
}
float I3f(float eta, float q, float R, float i4) {
  float yb = eta * u_cdip + q * u_sdip;
  float db = eta * u_sdip - q * u_cdip;
  if (u_vertical == 0) {
    return (1.0 - 2.0 * u_nu) * (yb / (u_cdip * (R + db)) - log(R + eta)) +
      u_sdip / u_cdip * i4;
  }
  return (1.0 - 2.0 * u_nu) / 2.0 *
    (eta / (R + db) + yb * q / ((R + db) * (R + db)) - log(R + eta));
}
float I1f(float xi, float eta, float q, float R, float i5) {
  float db = eta * u_sdip - q * u_cdip;
  if (u_vertical == 0) {
    return (1.0 - 2.0 * u_nu) * (-xi / (u_cdip * (R + db))) -
      u_sdip / u_cdip * i5;
  }
  return -(1.0 - 2.0 * u_nu) / 2.0 * xi * q / ((R + db) * (R + db));
}

struct Sub { vec3 ss; vec3 ds; vec3 tf; };

// All nine displacement subfunctions at one Chinnery corner (xi, eta, q).
Sub subAt(float xi, float eta, float q) {
  float sd = u_sdip, cd = u_cdip;
  float R = sqrt(xi * xi + eta * eta + q * q);
  float db = eta * sd - q * cd;
  float yb = eta * cd + q * sd;
  float Re = R + eta;
  float Rx = R + xi;
  float at = atan(xi * eta / (q * R));

  float i5 = I5f(xi, eta, q, R, db);
  float i4 = I4f(db, eta, q, R);
  float i3 = I3f(eta, q, R, i4);
  float i2 = (1.0 - 2.0 * u_nu) * (-log(Re)) - i3;
  float i1 = I1f(xi, eta, q, R, i5);

  Sub s;
  // strike-slip [eq (25)]
  s.ss = vec3(
    xi * q / (R * Re) + at + i1 * sd,
    (eta * cd + q * sd) * q / (R * Re) + q * cd / Re + i2 * sd,
    db * q / (R * Re) + q * sd / Re + i4 * sd);
  // dip-slip [eq (26)]
  s.ds = vec3(
    q / R - i3 * sd * cd,
    (eta * cd + q * sd) * q / (R * Rx) + cd * at - i1 * sd * cd,
    db * q / (R * Rx) + sd * at - i5 * sd * cd);
  // tensile [eq (27)]
  s.tf = vec3(
    q * q / (R * Re) - i3 * sd * sd,
    -db * q / (R * Rx) - sd * (xi * q / (R * Re) - at) - i1 * sd * sd,
    (eta * cd + q * sd) * q / (R * Rx) + cd * (xi * q / (R * Re) - at) -
      i5 * sd * sd);
  return s;
}

// Centroid (e, n) km -> Okada reference (x, p, q). Shared by all quantities.
vec3 okadaXPQ(float e, float n) {
  float d = u_depth + u_sdip * u_W * 0.5;
  float ec = e + u_cstr * u_cdip * u_W * 0.5;
  float nc = n - u_sstr * u_cdip * u_W * 0.5;
  float x = u_cstr * nc + u_sstr * ec + u_L * 0.5;
  float y = u_sstr * nc - u_cstr * ec + u_cdip * u_W;
  float p = y * u_cdip + d * u_sdip;
  float q = y * u_sdip - d * u_cdip;
  if (abs(q) < 1.0e-12) q = 1.0e-12; // avoid 0/0 in the atan term on q=0 plane
  return vec3(x, p, q);
}

// Returns surface displacement (ue, un, uz) at east/north offset (e, n) km.
vec3 okadaDisp(float e, float n) {
  vec3 xpq = okadaXPQ(e, n);
  float x = xpq.x, p = xpq.y, q = xpq.z;

  Sub a = subAt(x, p, q);
  Sub b = subAt(x, p - u_W, q);
  Sub c = subAt(x - u_L, p, q);
  Sub dd = subAt(x - u_L, p - u_W, q);
  vec3 ss = a.ss - b.ss - c.ss + dd.ss;
  vec3 ds = a.ds - b.ds - c.ds + dd.ds;
  vec3 tf = a.tf - b.tf - c.tf + dd.tf;

  vec3 u = -u_U1s * ss - u_U2s * ds + u_U3s * tf; // (ux, uy, uz)
  return vec3(u_sstr * u.x - u_cstr * u.y,   // ue
              u_cstr * u.x + u_sstr * u.y,   // un
              u.z);                          // uz
}

// --- Tilt + strain shared subfunctions [Okada eqs (40)-(41), (34)-(35)] ---
float Af(float x, float R) {
  float Rx = R + x;
  return (2.0 * R + x) / (R * R * R * Rx * Rx);
}
float K1f(float xi, float eta, float q, float R) {
  float db = eta * u_sdip - q * u_cdip;
  if (u_vertical == 0) {
    return (1.0 - 2.0 * u_nu) * xi / u_cdip *
      (1.0 / (R * (R + db)) - u_sdip / (R * (R + eta)));
  }
  return (1.0 - 2.0 * u_nu) * xi * q / ((R + db) * (R + db));
}
float K3f(float xi, float eta, float q, float R) {
  float db = eta * u_sdip - q * u_cdip;
  float yb = eta * u_cdip + q * u_sdip;
  if (u_vertical == 0) {
    return (1.0 - 2.0 * u_nu) / u_cdip *
      (q / (R * (R + eta)) - yb / (R * (R + db)));
  }
  return (1.0 - 2.0 * u_nu) * u_sdip / (R + db) * (xi * xi / (R * (R + db)) - 1.0);
}
float K2f(float eta, float q, float R, float k3) {
  return (1.0 - 2.0 * u_nu) * (-u_sdip / R + q * u_cdip / (R * (R + eta))) - k3;
}
float J1f(float xi, float eta, float q, float R, float k3) {
  float db = eta * u_sdip - q * u_cdip;
  if (u_vertical == 0) {
    return (1.0 - 2.0 * u_nu) / u_cdip *
      (xi * xi / (R * (R + db) * (R + db)) - 1.0 / (R + db)) -
      u_sdip / u_cdip * k3;
  }
  return (1.0 - 2.0 * u_nu) / 2.0 * q / ((R + db) * (R + db)) *
    (2.0 * xi * xi / (R * (R + db)) - 1.0);
}
float J2f(float xi, float eta, float q, float R, float k1) {
  float db = eta * u_sdip - q * u_cdip;
  float yb = eta * u_cdip + q * u_sdip;
  if (u_vertical == 0) {
    return (1.0 - 2.0 * u_nu) / u_cdip * xi * yb / (R * (R + db) * (R + db)) -
      u_sdip / u_cdip * k1;
  }
  return (1.0 - 2.0 * u_nu) / 2.0 * xi * u_sdip / ((R + db) * (R + db)) *
    (2.0 * q * q / (R * (R + db)) - 1.0);
}
float J3f(float xi, float eta, float q, float R, float j2) {
  return (1.0 - 2.0 * u_nu) * -xi / (R * (R + eta)) - j2;
}
float J4f(float eta, float q, float R, float j1) {
  return (1.0 - 2.0 * u_nu) * (-u_cdip / R - q * u_sdip / (R * (R + eta))) - j1;
}

// Tilt subfunctions at one Chinnery corner: vec2 = (uzx, uzy) per mechanism.
struct TiltSub { vec2 ss; vec2 ds; vec2 tf; };
TiltSub tiltAt(float xi, float eta, float q) {
  float sd = u_sdip, cd = u_cdip;
  float R = sqrt(xi * xi + eta * eta + q * q);
  float R3 = R * R * R;
  float db = eta * sd - q * cd;
  float yb = eta * cd + q * sd;
  float Re = R + eta, Rx = R + xi;
  float ae = Af(eta, R), ax = Af(xi, R);
  float k1 = K1f(xi, eta, q, R);
  float k3 = K3f(xi, eta, q, R);
  float k2 = K2f(eta, q, R, k3);

  TiltSub t;
  t.ss = vec2(                                       // [eq (37)]
    -xi * q * q * ae * cd + (xi * q / R3 - k1) * sd,
    db * q / R3 * cd + (xi * xi * q * ae * cd - sd / R + yb * q / R3 - k2) * sd);
  t.ds = vec2(                                       // [eq (38)]
    db * q / R3 + q * sd / (R * Re) + k3 * sd * cd,
    yb * db * q * ax - (2.0 * db / (R * Rx) + xi * sd / (R * Re)) * sd +
      k1 * sd * cd);
  t.tf = vec2(                                       // [eq (39)]
    q * q / R3 * sd - q * q * q * ae * cd + k3 * sd * sd,
    (yb * sd + db * cd) * q * q * ax + xi * q * q * ae * sd * cd -
      (2.0 * q / (R * Rx) - k1) * sd * sd);
  return t;
}

// Returns surface tilt (uze, uzn) at east/north offset (e, n) km.
vec2 okadaTilt(float e, float n) {
  vec3 xpq = okadaXPQ(e, n);
  float x = xpq.x, p = xpq.y, q = xpq.z;
  TiltSub a = tiltAt(x, p, q);
  TiltSub b = tiltAt(x, p - u_W, q);
  TiltSub c = tiltAt(x - u_L, p, q);
  TiltSub dd = tiltAt(x - u_L, p - u_W, q);
  vec2 ss = a.ss - b.ss - c.ss + dd.ss;
  vec2 ds = a.ds - b.ds - c.ds + dd.ds;
  vec2 tf = a.tf - b.tf - c.tf + dd.tf;
  vec2 uz = -u_U1s * ss - u_U2s * ds + u_U3s * tf; // (uzx, uzy)
  return vec2(-u_sstr * uz.x + u_cstr * uz.y,   // uze
              -u_cstr * uz.x - u_sstr * uz.y);  // uzn
}

// Strain subfunctions at one Chinnery corner: vec4 = (uxx,uxy,uyx,uyy) per mech.
struct StrainSub { vec4 ss; vec4 ds; vec4 tf; };
StrainSub strainAt(float xi, float eta, float q) {
  float sd = u_sdip, cd = u_cdip;
  float R = sqrt(xi * xi + eta * eta + q * q);
  float R3 = R * R * R;
  float db = eta * sd - q * cd;
  float yb = eta * cd + q * sd;
  float Re = R + eta, Rx = R + xi;
  float ae = Af(eta, R), ax = Af(xi, R);
  float k1 = K1f(xi, eta, q, R);
  float k3 = K3f(xi, eta, q, R);
  float j1 = J1f(xi, eta, q, R, k3);
  float j2 = J2f(xi, eta, q, R, k1);
  float j3 = J3f(xi, eta, q, R, j2);
  float j4 = J4f(eta, q, R, j1);

  StrainSub s;
  s.ss = vec4(                                       // [eq (31)]
    xi * xi * q * ae - j1 * sd,
    xi * xi * xi * db / (R3 * (eta * eta + q * q)) -
      (xi * xi * xi * ae + j2) * sd,
    xi * q / R3 * cd + (xi * q * q * ae - j2) * sd,
    yb * q / R3 * cd + (q * q * q * ae * sd - 2.0 * q * sd / (R * Re) -
      (xi * xi + eta * eta) / R3 * cd - j4) * sd);
  s.ds = vec4(                                       // [eq (32)]
    xi * q / R3 + j3 * sd * cd,
    yb * q / R3 - sd / R + j1 * sd * cd,
    yb * q / R3 + q * cd / (R * Re) + j1 * sd * cd,
    yb * yb * q * ax - (2.0 * yb / (R * Rx) + xi * cd / (R * Re)) * sd +
      j2 * sd * cd);
  s.tf = vec4(                                       // [eq (33)]
    xi * q * q * ae + j3 * sd * sd,
    -db * q / R3 - xi * xi * q * ae * sd + j1 * sd * sd,
    q * q / R3 * cd + q * q * q * ae * sd + j1 * sd * sd,
    (yb * cd - db * sd) * q * q * ax - q * (2.0 * sd * cd) / (R * Rx) -
      (xi * q * q * ae - j2) * sd * sd);
  return s;
}

// Returns surface strain (unn, une, uen, uee); positive = compression.
vec4 okadaStrain(float e, float n) {
  vec3 xpq = okadaXPQ(e, n);
  float x = xpq.x, p = xpq.y, q = xpq.z;
  StrainSub a = strainAt(x, p, q);
  StrainSub b = strainAt(x, p - u_W, q);
  StrainSub c = strainAt(x - u_L, p, q);
  StrainSub dd = strainAt(x - u_L, p - u_W, q);
  vec4 ss = a.ss - b.ss - c.ss + dd.ss;
  vec4 ds = a.ds - b.ds - c.ds + dd.ds;
  vec4 tf = a.tf - b.tf - c.tf + dd.tf;
  vec4 u = -u_U1s * ss - u_U2s * ds + u_U3s * tf; // (uxx, uxy, uyx, uyy)

  float s2 = 2.0 * u_sstr * u_cstr;          // sin(2*strike)
  float c2 = u_cstr * u_cstr, ss2 = u_sstr * u_sstr;
  float unn = c2 * u.x + s2 * (u.y + u.z) * 0.5 + ss2 * u.w;
  float une = ss2 * u.z + s2 * (u.x - u.w) * 0.5 - c2 * u.y;
  float uen = -c2 * u.z + s2 * (u.x - u.w) * 0.5 + ss2 * u.y;
  float uee = ss2 * u.x - s2 * (u.z + u.y) * 0.5 + c2 * u.w;
  return vec4(unn, une, uen, uee);
}

// Cyclic colormap for wrapped fringes.
vec3 fringeColor(float t) {
  float a = 6.28318530718 * t;
  return 0.5 + 0.5 * vec3(cos(a), cos(a - 2.09439510239), cos(a - 4.18879020479));
}

// Diverging blue-white-red colormap for signed amplitude, x in [-1, 1].
vec3 divergingColor(float x) {
  x = clamp(x, -1.0, 1.0);
  if (x < 0.0) return mix(vec3(1.0), vec3(0.23, 0.30, 0.75), -x);
  return mix(vec3(1.0), vec3(0.71, 0.09, 0.16), x);
}

// Distance from point p to segment a-b.
float segDist(vec2 p, vec2 a, vec2 b) {
  vec2 pa = p - a, ba = b - a;
  float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-12), 0.0, 1.0);
  return length(pa - ba * h);
}

// Draw the fault-outline overlay (white top edge, dark remaining edges).
vec3 drawFault(vec3 col, float e, float n) {
  if (u_showFault != 1) return col;
  vec2 P = vec2(e, n);
  float dTop = segDist(P, u_c0, u_c1);             // top edge = fault trace
  float dRest = min(min(segDist(P, u_c1, u_c2), segDist(P, u_c2, u_c3)),
                    segDist(P, u_c3, u_c0));
  float aa = u_kmPerPx;
  float rest = 1.0 - smoothstep(1.5 * aa, 2.5 * aa, dRest);
  float top = 1.0 - smoothstep(2.5 * aa, 3.5 * aa, dTop);
  col = mix(col, vec3(0.05), rest * 0.85);
  return mix(col, vec3(1.0), top);
}

void main() {
  vec2 ndc = gl_FragCoord.xy / u_resolution;       // 0..1, pixel centers
  float e = u_origin.x + (ndc.x * 2.0 - 1.0) * u_extent.x;
  float n = u_origin.y + (ndc.y * 2.0 - 1.0) * u_extent.y;

  // Okada (1985) is singular along the fault's surface trace, where a buried
  // dislocation projects to z=0: denominators vanish and the field blows up
  // (and can produce NaN/Inf). Flag pixels whose value is unphysically large --
  // the "!(mag < T)" form is also true for NaN -- and paint them neutral gray
  // instead of letting the colormap speckle. The buried-fault constraint keeps
  // the trace off-grid; this is the backstop.
  vec3 col;
  bool singular;
  float val;

  if (u_quantity == 1) {                           // tilt (uze, uzn)
    vec2 t = okadaTilt(e, n);
    if (u_mode == 1) { outColor = vec4(t, 0.0, 0.0); return; }
    val = (u_view == 0) ? t.x : t.y;
    singular = !(max(abs(t.x), abs(t.y)) < 1.0);   // > 1 rad, or NaN/Inf
    col = divergingColor(val / u_ampScale);
  } else if (u_quantity == 2) {                    // strain (unn,une,uen,uee)
    vec4 s = okadaStrain(e, n);
    if (u_mode == 1) { outColor = s; return; }
    val = (u_view == 0) ? s.w                      // Eee = uee
        : (u_view == 1) ? s.x                      // Enn = unn
        : (u_view == 2) ? 0.5 * (s.y + s.z)        // Ene = shear
        : (s.w + s.x);                             // areal = uee + unn
    singular = !(max(abs(s.x), abs(s.w)) < 1.0);   // > 100% strain, or NaN/Inf
    col = divergingColor(val / u_ampScale);
  } else {                                         // displacement
    vec3 disp = okadaDisp(e, n);
    float los = dot(disp, u_look);
    if (u_mode == 1) { outColor = vec4(disp, los); return; }
    singular = !(max(max(abs(disp.x), abs(disp.y)), abs(disp.z)) < 0.02); // >20 m
    if (u_view == 0) {
      col = fringeColor(fract(los / u_fringe));    // wrapped interferogram
    } else {
      val = (u_view == 1) ? los
          : (u_view == 2) ? disp.x
          : (u_view == 3) ? disp.y : disp.z;
      col = divergingColor(val / u_ampScale);      // signed amplitude
    }
  }
  if (singular) col = vec3(0.45);

  outColor = vec4(drawFault(col, e, n), 1.0);
}`;
