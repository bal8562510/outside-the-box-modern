import csv
import time
import numpy as np

from model import MNISTCNY19
from compare_with_baseline import load_keras_weights
from run_baseline import load, extract, take, stats, LAYERS, DEVICE, seed_all
from monitor_rotated import RotatedBox, LayerMonitor, Box, cluster_refinement

# =====================================================================
# BOX / ROTATED BOX / ELLIPSOID / ROTATED ELLIPSOID / INTERSECTIONS
# Data: full fit data ("raw") and outlier-removed fit data ("_clean")
#
# NO LEAKAGE: outlier removal, k-means, every shape (box, ellipsoid, radius)
# are fitted on the FIT part only. Epsilon is chosen on VALIDATION only.
# TEST is only used for reporting.
# =====================================================================
MODELS_DIR = r"C:\work\original\models"
THRESHOLD = 0.07

PRESET = "final"                  # "quick_test" (seed 0, n=3)  or  "final" (seeds 0-2, n=2..9)
PRESETS = {"quick_test": dict(seeds=[0], n_range=[3]),
           "final":      dict(seeds=[0, 1, 2], n_range=list(range(2, 10)))}
SEEDS, N_RANGE = PRESETS[PRESET]["seeds"], PRESETS[PRESET]["n_range"]
OUT_FILE = f"results_ellipsoid_intersection_mnist_{PRESET}.csv"

VAL_FRAC = 0.2
TARGET_FW = 0.05                  # selected = strictest epsilon with validation FW <= 5%
EPSILONS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0, 1.5]   # box, rotated, intersections
EPS_ELL = [-0.06, -0.05, -0.04, -0.03, -0.025, -0.02, -0.015, -0.01, -0.0075,
           -0.005, -0.0025, -0.001, 0.0, 0.05, 0.1, 0.2, 0.5]                       # ellipsoids only (fine grid near 0)
ELLIPSOID_SHAPES = ("ellipsoid", "rotated_ellipsoid")
MATCH_FW = [3, 5, 8]
MAX_GAP = 5.0                     # matched-FW is only computed if the 2 curve points are <= 5 FW points apart
OUTLIER_FRAC = 0.01               # drop the 1% farthest fit points per class and layer
MIN_SD = 1e-6


# =====================================================================
# SHAPES
# =====================================================================
class Ellipsoid:
    """Axis-aligned ellipsoid: centre = mean, axes ~ per-dimension std.
    Radius = largest scaled distance of a fit point. epsilon: radius * (1 + eps)."""
    TOL = 1e-6

    def __init__(self, pts):
        pts = np.asarray(pts, dtype=np.float64)
        self.mu = pts.mean(0)
        self.sd = np.maximum(pts.std(0), MIN_SD)
        self.r = self._dist(pts).max()

    def _dist(self, X):
        return np.sqrt((((np.asarray(X, dtype=np.float64) - self.mu) / self.sd) ** 2).sum(1))

    def contains(self, X, eps=0.0):
        return self._dist(X) <= self.r * (1.0 + eps) + self.TOL


class RotatedEllipsoid(Ellipsoid):
    """Mahalanobis ellipsoid (same principal axes as RotatedBox, full basis)."""
    def __init__(self, pts):
        pts = np.asarray(pts, dtype=np.float64)
        self.mu = pts.mean(0)
        cen = pts - self.mu
        vals, self.R = np.linalg.eigh(cen.T @ cen / len(pts))
        self.sd = np.maximum(np.sqrt(np.maximum(vals, 0.0)), MIN_SD)
        self.r = self._dist(pts).max()

    def _dist(self, X):
        z = (np.asarray(X, dtype=np.float64) - self.mu) @ self.R
        return np.sqrt(((z / self.sd) ** 2).sum(1))


class BoxEllipsoid:
    """INTERSECTION: a point is accepted only if it is inside the box AND inside the ellipsoid.
    The ellipsoid keeps its fit radius (eps = 0, all fit points inside); epsilon only loosens the box."""
    def __init__(self, pts):
        self.box, self.ell = Box(pts), Ellipsoid(pts)

    def contains(self, X, eps=0.0):
        return self.box.contains(X, eps) & self.ell.contains(X, 0.0)


class RotatedBoxEllipsoid(BoxEllipsoid):
    """Same idea with the rotated box and the rotated ellipsoid."""
    def __init__(self, pts):
        self.box, self.ell = RotatedBox(pts), RotatedEllipsoid(pts)


SHAPES = {"box": Box, "rotated": RotatedBox, "ellipsoid": Ellipsoid, "rotated_ellipsoid": RotatedEllipsoid,
          "box_ell": BoxEllipsoid, "rot_box_ell": RotatedBoxEllipsoid}

V = {}
for shape in SHAPES:
    V[shape] = dict(shape=shape, clean=False)
    V[shape + "_clean"] = dict(shape=shape, clean=True)

EFFECTS = [   # (label, variant, reference): change in detection at matched FW = 5% (positive = variant better)
    ("outliers: box_ell", "box_ell_clean", "box_ell"),
    ("outliers: rot_box_ell", "rot_box_ell_clean", "rot_box_ell"),
    ("box_ell vs box (raw)", "box_ell", "box"),
    ("box_ell vs box (clean)", "box_ell_clean", "box_clean"),
    ("box_ell vs ellipsoid (raw)", "box_ell", "ellipsoid"),
    ("box_ell vs ellipsoid (clean)", "box_ell_clean", "ellipsoid_clean"),
    ("rot_box_ell vs rotated (raw)", "rot_box_ell", "rotated"),
    ("rot_box_ell vs rotated (clean)", "rot_box_ell_clean", "rotated_clean"),
    ("rot_box_ell vs rot_ellipsoid (raw)", "rot_box_ell", "rotated_ellipsoid"),
    ("rot_box_ell vs rot_ellipsoid (clean)", "rot_box_ell_clean", "rotated_ellipsoid_clean"),
    ("outliers: box", "box_clean", "box"),
    ("outliers: rotated", "rotated_clean", "rotated"),
    ("outliers: ellipsoid", "ellipsoid_clean", "ellipsoid"),
    ("outliers: rotated ellipsoid", "rotated_ellipsoid_clean", "rotated_ellipsoid"),
    ("rotated vs box (clean)", "rotated_clean", "box_clean"),
    ("ellipsoid vs box (clean)", "ellipsoid_clean", "box_clean"),
    ("rot. ellipsoid vs rotated (clean)", "rotated_ellipsoid_clean", "rotated_clean"),
]


# =====================================================================
# OUTLIER REMOVAL (fit part only)
# =====================================================================
def remove_outliers(Vd, y, n):
    keep = np.ones(len(y), dtype=bool)
    for l in LAYERS:
        for c in range(n):
            idx = np.where(y == c)[0]
            if len(idx) < 20:
                continue
            X = Vd[l][idx]
            d = (((X - np.median(X, 0)) / (X.std(0) + 1e-9)) ** 2).mean(1)
            keep[idx[d > np.quantile(d, 1 - OUTLIER_FRAC)]] = False
    return keep


# =====================================================================
# MONITORS (k-means cached per data version -> all shapes use the SAME clusters)
# =====================================================================
CLUSTERS = {}


class CachedLayerMonitor(LayerMonitor):
    def fit(self, Vl, y, n_classes, key):
        rng = np.random.default_rng(0)
        for c in range(n_classes):
            pts = Vl[y == c]
            if len(pts) == 0:
                self.shapes[c] = []
                continue
            if (key, c) not in CLUSTERS:
                CLUSTERS[(key, c)] = cluster_refinement(pts, self.threshold).predict(pts)
            lab = CLUSTERS[(key, c)]
            self.shapes[c] = [self._make_shape(pts[lab == j], rng) for j in np.unique(lab)]
        return self


class ShapeMonitor:
    """One shape per cluster, per layer; every layer must accept. Layer -1 is always a plain Box."""
    def __init__(self, shape_cls, clean):
        self.shape_cls, self.clean = shape_cls, clean

    def fit(self, Vd, y, n):
        self.mons = {}
        for l in LAYERS:
            cls, key = (Box, ("L1", self.clean)) if l == -1 else (self.shape_cls, (self.clean, l))
            self.mons[l] = CachedLayerMonitor(shape_cls=cls, threshold=THRESHOLD).fit(Vd[l], y, n, key)
        return self

    def n_clusters(self):
        return sum(len(s) for m in self.mons.values() for s in m.shapes.values())

    def accepts(self, Vd, preds, eps):
        ok = np.ones(len(preds), dtype=bool)
        for l, m in self.mons.items():
            m.epsilon = eps
            ok &= m.accepts(Vd[l], preds)
        return ok


# =====================================================================
# REPORTING
# =====================================================================
def test_rates(r):
    return (100 * r["nov_det"] / (r["nov_det"] + r["nov_undet"]),
            100 * r["fp"] / (r["fp"] + r["tn"]))


def det_at_fw(g, point):
    """Detection % at a test FW %, interpolated along ONE run's curve.
    NaN if the point is outside the curve OR the two neighbouring curve points are > MAX_GAP FW points apart (cliff)."""
    pts = sorted(test_rates(r)[::-1] for r in g)       # (fw, det)
    fw, det = [p[0] for p in pts], [p[1] for p in pts]
    if not (fw[0] <= point <= fw[-1]):
        return np.nan
    i = int(np.searchsorted(fw, point))
    lo, hi = max(i - 1, 0), i
    if fw[hi] - fw[lo] > MAX_GAP:
        return np.nan
    return float(np.interp(point, fw, det))


def matched(rs, point):
    groups = {}
    for r in rs:
        groups.setdefault((r["method"], r["n_classes"], r["seed"]), []).append(r)
    return {k: det_at_fw(g, point) for k, g in groups.items()}


def print_table(rs):
    print(f"{'method':28s}{'eps':>7s}{'valFW%':>8s}{'det%':>8s}{'FW%':>8s}{'clusters':>10s}{'reached':>9s}")
    for label in dict.fromkeys(r["method"] for r in rs):
        g = [r for r in rs if r["method"] == label]
        t = np.array([test_rates(r) for r in g])
        print(f"{label:28s}{np.mean([r['param'] for r in g]):7.3f}{np.mean([r['val_fw_pct'] for r in g]):8.1f}"
              f"{t[:, 0].mean():8.1f}{t[:, 1].mean():8.1f}{np.mean([r['n_clusters'] for r in g]):10.1f}"
              f"{sum(r['target_reached'] for r in g):5d}/{len(g):<3d}")


def print_matched(rs):
    maps = {p: matched(rs, p) for p in MATCH_FW}
    print(f"{'method':28s}" + "".join(f"{'det% at FW=' + str(p) + '%':>16s}" for p in MATCH_FW))
    for label in dict.fromkeys(r["method"] for r in rs):
        line = f"{label:28s}"
        for p in MATCH_FW:
            v = [x for (m, _, _), x in maps[p].items() if m == label and not np.isnan(x)]
            line += f"{np.mean(v):9.1f} ({len(v):2d})  " if v else f"{'n/a':>16s}"
        print(line)


def print_effects(rs):
    d = matched(rs, 5)
    print(f"{'comparison':40s}{'change in det% at FW=5%':>30s}")
    for label, a, b in EFFECTS:
        diffs = [d[(a, n, s)] - d[(b, n, s)] for (m, n, s) in d
                 if m == a and (b, n, s) in d and not np.isnan(d[(a, n, s)]) and not np.isnan(d[(b, n, s)])]
        if diffs:
            print(f"{label:40s}{np.mean(diffs):+14.1f} points  ({len(diffs)} runs)")


def report(rs, title):
    print(f"\n================ {title} ================")
    print("A) validation-selected epsilon = HEADLINE numbers (reached = runs that hit the 5% FW target)")
    print_table([r for r in rs if r["selected"]])
    print("\nB) matched false warnings: detection interpolated at test FW = 3 / 5 / 8 % (exploratory; (k) = valid runs)")
    print_matched(rs)
    print("\nC) effect at matched FW = 5% (positive = first variant is better)")
    print_effects(rs)


def save():
    with open(OUT_FILE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)


# =====================================================================
# EXPERIMENT
# =====================================================================
print(f"PRESET = {PRESET} | seeds = {SEEDS} | n = {N_RANGE} | {len(V)} variants")
print(f"outliers {100 * OUTLIER_FRAC:.0f}% | selection: strictest epsilon with validation FW <= {100 * TARGET_FW:.0f}%")
x_tr, y_tr, x_te, y_te = load()
rows = []

for n in N_RANGE:
    model = MNISTCNY19(n)
    load_keras_weights(model, rf"{MODELS_DIR}\CNY19id1_MNIST_0-{n-1}.h5")
    model.to(DEVICE).eval()

    tr, te = y_tr < n, y_te < n
    V_tr = extract(model, x_tr[tr]); gt_tr = y_tr[tr]
    ok = V_tr[-1].argmax(1) == gt_tr
    V_ok, y_ok = take(V_tr, ok), gt_tr[ok]
    V_run = extract(model, x_te)
    pred = V_run[-1].argmax(1)
    correct = pred == y_te
    print(f"\nn={n}: accuracy on known digits = {correct[te].mean():.4f}")

    for seed in SEEDS:
        is_val = np.zeros(len(y_ok), dtype=bool)
        is_val[np.random.default_rng(seed).permutation(len(y_ok))[:int(VAL_FRAC * len(y_ok))]] = True
        V_fit, y_fit = take(V_ok, ~is_val), y_ok[~is_val]
        V_val, y_val = take(V_ok, is_val), y_ok[is_val]

        keep = remove_outliers(V_fit, y_fit, n)              # fit part only
        V_cln, y_cln = take(V_fit, keep), y_fit[keep]
        n_removed = int((~keep).sum())
        print(f"   n={n} seed={seed}: removed {n_removed} of {len(keep)} fit points")
        CLUSTERS.clear()

        for name, v in V.items():
            Vf, yf = (V_cln, y_cln) if v["clean"] else (V_fit, y_fit)
            seed_all(seed)
            mon = ShapeMonitor(SHAPES[v["shape"]], v["clean"])
            t0 = time.perf_counter()
            mon.fit(Vf, yf, n)
            fit_s = time.perf_counter() - t0

            ncl = mon.n_clusters()
            base = dict(seed=seed, n_classes=n, method=name, shape=v["shape"], clean=v["clean"],
                        n_clusters=ncl, n_fit=len(yf), n_removed=n_removed if v["clean"] else 0,
                        fit_s=round(fit_s, 2), param_type="epsilon")

            grid = EPS_ELL if v["shape"] in ELLIPSOID_SHAPES else EPSILONS
            mine, chosen = [], None
            for eps in grid:
                val_fw = 1 - mon.accepts(V_val, y_val, eps).mean()
                if chosen is None and val_fw <= TARGET_FW:
                    chosen = eps                             # selection uses validation only
                acc = mon.accepts(V_run, pred, eps)
                row = dict(base, param=eps, val_fw_pct=round(100 * val_fw, 2))
                row.update(stats(correct, acc, y_te, n))
                mine.append(row)
            reached = chosen is not None
            if not reached:
                chosen = grid[-1]                            # loosest value
            for r in mine:
                r["selected"] = (r["param"] == chosen)
                r["target_reached"] = reached
            rows.extend(mine)

            s = next(r for r in mine if r["selected"])
            det, fw = test_rates(s)
            print(f"   n={n} seed={seed} {name:26s} eps={chosen:<7} valFW={s['val_fw_pct']:4.1f}% | "
                  f"test det {det:5.1f}%  FW {fw:4.1f}% | fit {fit_s:.0f}s"
                  + ("" if reached else "  (target FW not reached)"))

    report([r for r in rows if r["n_classes"] == n], f"n={n}")
    save()

report(rows, "OVERALL (all n, all seeds)")
print("saved", OUT_FILE)