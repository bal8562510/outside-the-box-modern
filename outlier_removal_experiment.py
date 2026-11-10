import csv
import time
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.svm import OneClassSVM
from sklearn.decomposition import KernelPCA

from model import MNISTCNY19
from compare_with_baseline import load_keras_weights
from run_baseline import load, extract, take, stats, LAYERS, DEVICE, seed_all
from monitor_rotated import RotatedBox, LayerMonitor, Box, cluster_refinement

# =====================================================================
# ALL METHODS + ALL FIXES IN ONE RUN (MNIST novelty-detection monitors)
# Methods: box, rotated, adaptive, RFF(box/rotated), kernel PCA + box, one-class SVM
# Fixes:   1 outliers removed | 2 feature scaling | 3 side floor | 4 percentile boxes / quantile SVM threshold
#          5 few points per cluster: PCA-95 rotated box, kPCA with 10 components
# LEAKAGE: every fix and map is fitted on the FIT part only; epsilon / q is chosen on VALIDATION only.
# =====================================================================
MODELS_DIR = r"C:\work\original\models"
THRESHOLD = 0.07

PRESET = "final"                  # "quick_test" (seed 0, n=3)  or  "final" (seeds 0-2, n=2..9)
PRESETS = {"quick_test": dict(seeds=[0], n_range=[3]),
           "final":      dict(seeds=[0, 1, 2], n_range=list(range(2, 10)))}
SEEDS, N_RANGE = PRESETS[PRESET]["seeds"], PRESETS[PRESET]["n_range"]
OUT_FILE = f"results_all_methods_and_fixes_outliers_floor_scaling_pca_rff_kpca_ocsvm_mnist_{PRESET}.csv"
RUN_ONLY = None                        # e.g. ["box", "rotated_clean"] runs only variants whose name is in the list

VAL_FRAC = 0.2
TARGET_FW = 0.05                       # selected = strictest value with validation FW <= 5%
EPSILONS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0, 1.5]     # boxes (loose = large)
Q_GRID = [0.30, 0.20, 0.10, 0.07, 0.05, 0.04, 0.03, 0.025, 0.02, 0.015, 0.01, 0.0075,
          0.005, 0.0025, 0.0]          # ocsvm: threshold = quantile of FIT scores (strict first; 0.0 = old s_min rule)
MATCH_FW = [3, 5, 8]                   # exploratory matched comparison (test curves, ranking only)

# ---- fixed settings, NOT tuned ----
OUTLIER_FRAC = 0.01                    # fix 1: drop the 1% farthest fit points per class and layer
FLOOR_FRAC = 0.1                       # fix 3: every side at least 0.1 * unit std wide
Q_LOW = 0.5                            # fix 4: percentile boxes use the 0.5% and 99.5% percentiles
PCA_VAR = 0.95                         # fix 5: rotated box only on axes covering 95% of the cluster variance
KPCA_K = 10                            # fix 5: kPCA with only 10 components
RFF_SCALE = 2.0                        # as in Experiment 3
SVM_NU = 0.1
KPCA_LANDMARKS = 3000
HIDDEN = [-2, -3, -4]                  # layer -1 is always a plain Box

# ---- variants: shape (box, rotated, adaptive, box_floor, rotated_floor, box_q, rotated_q, rotated_pca95, ocsvm) ----
V = {}
def add(name, shape, clean=False, tf="none", scaled=False, **kw):
    V[name] = dict(shape=shape, clean=clean, tf=tf, scaled=scaled, **kw)

for c, s in [(False, ""), (True, "_clean")]:
    add("box" + s, "box", c)
    add("rotated" + s, "rotated", c)
    add("adaptive_validation" + s, "adaptive", c)
    add("box" + s + "_floor", "box_floor", c)
    add("rotated" + s + "_floor", "rotated_floor", c)
add("box_q", "box_q")
add("rotated_q", "rotated_q")
add("rotated_pca95", "rotated_pca95")
add("rotated_clean_pca95", "rotated_pca95", True)
for m in (1, 4, 5):
    add(f"rff_d{m}", "box", tf="rff", mult=m)
add("rff_clean_d4", "box", True, "rff", mult=4)
for m in (1, 2, 4, 5):
    add(f"rff_rot_d{m}", "rotated", tf="rff", mult=m)
add("rff_rot_clean_d1", "rotated", True, "rff", mult=1)
add("rff_rot_pca95_d2", "rotated_pca95", tf="rff", mult=2)
add("rff_rot_pca95_d4", "rotated_pca95", tf="rff", mult=4)
add("kpca", "box", tf="kpca", gmul=1, k=0)
add("kpca_clean", "box", True, "kpca", gmul=1, k=0)
add("kpca_clean_gamma4", "box", True, "kpca", gmul=4, k=0)
add("kpca_clean_gamma16", "box", True, "kpca", gmul=16, k=0)
add("kpca_clean_scaled", "box", True, "kpca", scaled=True, gmul=1, k=0)
add("kpca_clean_k10", "box", True, "kpca", gmul=1, k=KPCA_K)
add("kpca_clean_k10_gamma4", "box", True, "kpca", gmul=4, k=KPCA_K)
add("kpca_clean_k10_gamma16", "box", True, "kpca", gmul=16, k=KPCA_K)
add("ocsvm", "ocsvm", gmul=1)
add("ocsvm_clean", "ocsvm", True, gmul=1)
add("ocsvm_clean_scaled", "ocsvm", True, scaled=True, gmul=1)
add("ocsvm_clean_scaled_gamma4", "ocsvm", True, scaled=True, gmul=4)
add("ocsvm_clean_scaled_gamma16", "ocsvm", True, scaled=True, gmul=16)
add("ocsvm_clean_gamma4", "ocsvm", True, gmul=4)
add("ocsvm_clean_gamma16", "ocsvm", True, gmul=16)
add("ocsvm_clean_gamma64", "ocsvm", True, gmul=64)
if RUN_ONLY:
    V = {k: v for k, v in V.items() if k in RUN_ONLY}

EFFECTS = [   # (label, variant, reference): change in detection at matched FW = 5% (positive = variant better)
    ("1 outliers: box", "box_clean", "box"),
    ("1 outliers: rotated", "rotated_clean", "rotated"),
    ("1 outliers: adaptive", "adaptive_validation_clean", "adaptive_validation"),
    ("1 outliers: rff_d4", "rff_clean_d4", "rff_d4"),
    ("1 outliers: rff_rot_d1", "rff_rot_clean_d1", "rff_rot_d1"),
    ("1 outliers: kpca", "kpca_clean", "kpca"),
    ("1 outliers: ocsvm", "ocsvm_clean", "ocsvm"),
    ("2 scaling: ocsvm", "ocsvm_clean_scaled", "ocsvm_clean"),
    ("2 scaling: kpca", "kpca_clean_scaled", "kpca_clean"),
    ("3 floor: box", "box_floor", "box"),
    ("3 floor: rotated", "rotated_floor", "rotated"),
    ("3 floor: box (clean)", "box_clean_floor", "box_clean"),
    ("3 floor: rotated (clean)", "rotated_clean_floor", "rotated_clean"),
    ("4 percentile box", "box_q", "box"),
    ("4 percentile rotated", "rotated_q", "rotated"),
    ("5 pca95 rotated", "rotated_pca95", "rotated"),
    ("5 pca95 rotated (clean)", "rotated_clean_pca95", "rotated_clean"),
    ("5 pca95: rff_rot_d2", "rff_rot_pca95_d2", "rff_rot_d2"),
    ("5 pca95: rff_rot_d4", "rff_rot_pca95_d4", "rff_rot_d4"),
    ("5 kpca 10 components", "kpca_clean_k10", "kpca_clean"),
    ("gamma: kpca x4", "kpca_clean_gamma4", "kpca_clean"),
    ("gamma: kpca x16", "kpca_clean_gamma16", "kpca_clean"),
    ("gamma: kpca k10 x4", "kpca_clean_k10_gamma4", "kpca_clean_k10"),
    ("gamma: kpca k10 x16", "kpca_clean_k10_gamma16", "kpca_clean_k10"),
    ("gamma: ocsvm x4", "ocsvm_clean_gamma4", "ocsvm_clean"),
    ("gamma: ocsvm x16", "ocsvm_clean_gamma16", "ocsvm_clean"),
    ("gamma: ocsvm x64", "ocsvm_clean_gamma64", "ocsvm_clean"),
    ("gamma: ocsvm scaled x4", "ocsvm_clean_scaled_gamma4", "ocsvm_clean_scaled"),
    ("gamma: ocsvm scaled x16", "ocsvm_clean_scaled_gamma16", "ocsvm_clean_scaled"),
    ("method: rotated vs box (raw)", "rotated", "box"),
    ("method: rotated vs box (clean)", "rotated_clean", "box_clean"),
    ("method: adaptive vs box", "adaptive_validation", "box"),
    ("method: rff_d4 vs box", "rff_d4", "box"),
    ("method: rff_rot_d1 vs rotated", "rff_rot_d1", "rotated"),
    ("method: kpca vs box (raw)", "kpca", "box"),
    ("method: kpca vs box (clean)", "kpca_clean", "box_clean"),
    ("method: ocsvm vs box (raw)", "ocsvm", "box"),
    ("method: ocsvm vs box (clean)", "ocsvm_clean", "box_clean"),
]


# =====================================================================
# RFF (your class)
# =====================================================================
class RFF:
    """Random Fourier Features: z(x) = sqrt(2/D) * cos(x W + b)
    W ~ N(0, 1/sigma^2), b ~ U[0, 2pi]. Fixed after fit(), so the SAME map is
    applied to the fit data and to the run-time data.
    sigma = median pairwise distance of (a subsample of) the FIT data * sigma_scale.
    """
    def __init__(self, D, sigma_scale=2.0, seed=0):
        self.D, self.sigma_scale, self.seed = D, sigma_scale, seed

    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        sub = X[rng.choice(len(X), min(500, len(X)), replace=False)].astype(np.float64)
        d = pairwise_distances(sub)
        med = np.median(d[np.triu_indices(len(sub), 1)])
        self.sigma = (med if med > 0 else 1.0) * self.sigma_scale
        self.W = rng.normal(0.0, 1.0 / self.sigma, size=(X.shape[1], self.D))
        self.b = rng.uniform(0.0, 2 * np.pi, size=self.D)
        return self

    def transform(self, X):
        return np.sqrt(2.0 / self.D) * np.cos(np.asarray(X, dtype=np.float64) @ self.W + self.b)


# =====================================================================
# SHAPES (fixes 3, 4, 5)
# =====================================================================
def floor_box(scale):
    class FBox(Box):
        def __init__(self, pts):
            super().__init__(pts)
            c, half = (self.low + self.high) / 2, np.maximum((self.high - self.low) / 2, FLOOR_FRAC * scale / 2)
            self.low, self.high = c - half, c + half
    return FBox


def floor_rot(scale):
    class FRot(RotatedBox):
        def __init__(self, pts):
            super().__init__(pts)
            c, half = (self.low + self.high) / 2, np.maximum((self.high - self.low) / 2, FLOOR_FRAC * scale.mean() / 2)
            self.low, self.high = c - half, c + half
    return FRot


class QBox(Box):
    """Box from percentiles instead of min / max (robust to single extreme points)."""
    def __init__(self, pts):
        self.low, self.high = np.percentile(pts, Q_LOW, 0), np.percentile(pts, 100 - Q_LOW, 0)


class QRot(RotatedBox):
    """Rotated box with percentile sides."""
    def __init__(self, pts):
        super().__init__(pts)
        z = (np.asarray(pts, dtype=np.float64) - self.mu) @ self.R
        self.low, self.high = np.percentile(z, Q_LOW, 0), np.percentile(z, 100 - Q_LOW, 0)


class PCARot(RotatedBox):
    """Rotated box on the leading axes only (PCA_VAR of the cluster variance); the near-empty axes are not checked."""
    def __init__(self, pts):
        pts = np.asarray(pts, dtype=np.float64)
        self.mu = pts.mean(0)
        cen = pts - self.mu
        vals, vecs = np.linalg.eigh(cen.T @ cen / len(pts))
        vals, vecs = vals[::-1], vecs[:, ::-1]
        tot = vals.sum()
        k = 1 if tot <= 0 else int(np.searchsorted(np.cumsum(vals) / tot, PCA_VAR)) + 1
        self.R = vecs[:, :k]
        z = cen @ self.R
        self.low, self.high = z.min(0), z.max(0)


def shape_class(shape, X):
    scale = X.std(0)
    return {"box": Box, "rotated": RotatedBox, "adaptive": "adaptive_validation",
            "box_floor": floor_box(scale), "rotated_floor": floor_rot(scale),
            "box_q": QBox, "rotated_q": QRot, "rotated_pca95": PCARot}[shape]


# =====================================================================
# FIXES / MAPS (all fitted on the FIT part only, then applied to val and test)
# =====================================================================
def remove_outliers(Vd, y, n):
    """Mask of fit points to KEEP: per layer and class drop the OUTLIER_FRAC farthest (mean squared z-score to the median)."""
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


def scale_apply(Vf, others):
    Vf, out = dict(Vf), [dict(o) for o in others]
    for l in HIDDEN:
        mu, sd = Vf[l].mean(0), Vf[l].std(0)
        sd[sd < 1e-6] = 1.0
        Vf[l] = (Vf[l] - mu) / sd
        for o in out:
            o[l] = (o[l] - mu) / sd
    return Vf, out


def rff_apply(Vf, others, mult, seed):
    """One RFF map per hidden layer, fitted on the FIT part; D = mult * layer size. Layer -1 untouched."""
    Vf, out = dict(Vf), [dict(o) for o in others]
    for l in HIDDEN:
        rff = RFF(D=mult * Vf[l].shape[1], sigma_scale=RFF_SCALE, seed=seed * 100 + abs(l)).fit(Vf[l])
        Vf[l] = rff.transform(Vf[l])
        for o in out:
            o[l] = rff.transform(o[l])
    return Vf, out


def kpca_apply(Vf, others, gmul, k, seed):
    """Kernel PCA per hidden layer on a subsample of the FIT part; k components (0 = layer dimension)."""
    Vf, out = dict(Vf), [dict(o) for o in others]
    tf = lambda kp, X: np.vstack([kp.transform(np.asarray(X[i:i + 5000], dtype=np.float64)) for i in range(0, len(X), 5000)])
    for l in HIDDEN:
        X = Vf[l]
        rng = np.random.default_rng(seed * 100 + abs(l))
        sub = np.asarray(X[rng.choice(len(X), min(KPCA_LANDMARKS, len(X)), replace=False)], dtype=np.float64)
        gamma = gmul / (sub.shape[1] * sub.var() + 1e-12)
        kp = KernelPCA(n_components=min(k or sub.shape[1], sub.shape[1]), kernel="rbf", gamma=gamma).fit(sub)
        Vf[l] = tf(kp, X)
        for o in out:
            o[l] = tf(kp, o[l])
    return Vf, out


# =====================================================================
# MONITORS (k-means labels are cached per data version, so all shapes of one version share the same clusters)
# =====================================================================
CLUSTERS = {}


class CachedLayerMonitor(LayerMonitor):
    def fit(self, Vl, y, n_classes, key):
        rng = np.random.default_rng(0)                       # same as LayerMonitor.fit
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


class BoxMonitor:
    """Shape per cluster, per layer; every layer must accept. Layer -1 is always a plain Box on the raw outputs."""
    def __init__(self, shape, ckey):
        self.shape, self.ckey = shape, ckey

    def fit(self, Vd, y, n):
        self.mons = {}
        for l in LAYERS:
            if l == -1:
                cls, key = Box, ("L1", self.ckey[0])
            else:
                cls, key = shape_class(self.shape, Vd[l]), (self.ckey, l)
            self.mons[l] = CachedLayerMonitor(shape_cls=cls, threshold=THRESHOLD).fit(Vd[l], y, n, key)
        return self

    def n_clusters(self):
        return sum(len(s) for m in self.mons.values() for s in m.shapes.values())

    def n_support(self):
        return 0

    def accepts(self, key, Vd, preds, eps):
        ok = np.ones(len(preds), dtype=bool)
        for l, m in self.mons.items():
            m.epsilon = eps
            ok &= m.accepts(Vd[l], preds)
        return ok


class SVMMonitor:
    """One RBF one-class SVM per (hidden layer, class); layer -1 = Box. A layer accepts if score >= quantile q of the FIT scores."""
    def __init__(self, gmul, ckey):
        self.gmul, self.ckey, self.cache = gmul, ckey, {}

    def fit(self, Vd, y, n):
        self.box1 = CachedLayerMonitor(shape_cls=Box, threshold=THRESHOLD).fit(Vd[-1], y, n, ("L1", self.ckey[0]))
        self.models = {l: {} for l in HIDDEN}
        for l in HIDDEN:
            for c in range(n):
                P = np.asarray(Vd[l][y == c], dtype=np.float64)
                if len(P) == 0:
                    continue
                gamma = self.gmul / (P.shape[1] * P.var() + 1e-12)       # gmul = 1 is sklearn's "scale"
                svm = OneClassSVM(kernel="rbf", gamma=gamma, nu=SVM_NU).fit(P)
                self.models[l][c] = (svm, svm.decision_function(P))
        return self

    def n_clusters(self):
        return 0

    def n_support(self):
        return sum(len(m[0].support_) for d in self.models.values() for m in d.values())

    def _score(self, key, l, X, preds):
        if (key, l) not in self.cache:
            s = np.full(len(preds), -np.inf)                             # class without a model: always rejected
            for c, (svm, _) in self.models[l].items():
                idx = preds == c
                if idx.any():
                    s[idx] = svm.decision_function(X[idx])
            self.cache[(key, l)] = s
        return self.cache[(key, l)]

    def accepts(self, key, Vd, preds, q):
        ok = np.asarray(self.box1.accepts(Vd[-1], preds), dtype=bool)
        for l in HIDDEN:
            thr = np.full(len(preds), np.inf)
            for c, (_, sf) in self.models[l].items():
                thr[preds == c] = np.quantile(sf, q)
            ok &= self._score(key, l, Vd[l], preds) >= thr
        return ok


# =====================================================================
# REPORTING
# =====================================================================
def test_rates(r):
    return (100 * r["nov_det"] / (r["nov_det"] + r["nov_undet"]),
            100 * r["fp"] / (r["fp"] + r["tn"]))


def det_at_fw(g, point):
    """Detection % at a test FW %: linear interpolation along ONE run's curve (nan if outside the curve)."""
    pts = sorted(test_rates(r)[::-1] for r in g)       # (fw, det)
    fw, det = [p[0] for p in pts], [p[1] for p in pts]
    return float(np.interp(point, fw, det)) if fw[0] <= point <= fw[-1] else np.nan


def matched(rs, point):
    groups = {}
    for r in rs:
        groups.setdefault((r["method"], r["n_classes"], r["seed"]), []).append(r)
    return {k: det_at_fw(g, point) for k, g in groups.items()}


def print_table(rs):
    print(f"{'method':30s}{'param':>7s}{'valFW%':>8s}{'det%':>8s}{'FW%':>8s}{'clusters':>10s}{'pts/cl':>8s}{'reached':>9s}")
    for label in dict.fromkeys(r["method"] for r in rs):
        g = [r for r in rs if r["method"] == label]
        t = np.array([test_rates(r) for r in g])
        print(f"{label:30s}{np.mean([r['param'] for r in g]):7.3f}{np.mean([r['val_fw_pct'] for r in g]):8.1f}"
              f"{t[:, 0].mean():8.1f}{t[:, 1].mean():8.1f}"
              f"{np.mean([r['n_clusters'] for r in g]):10.1f}{np.mean([r['pts_per_cluster'] for r in g]):8.1f}"
              f"{sum(r['target_reached'] for r in g):5d}/{len(g):<3d}")


def print_matched(rs):
    maps = {p: matched(rs, p) for p in MATCH_FW}
    print(f"{'method':30s}" + "".join(f"{'det% at FW=' + str(p) + '%':>16s}" for p in MATCH_FW))
    for label in dict.fromkeys(r["method"] for r in rs):
        line = f"{label:30s}"
        for p in MATCH_FW:
            v = [x for (m, _, _), x in maps[p].items() if m == label and not np.isnan(x)]
            line += f"{np.mean(v):9.1f} ({len(v):2d})  " if v else f"{'n/a':>16s}"
        print(line)


def print_effects(rs):
    d = matched(rs, 5)
    print(f"{'comparison':34s}{'change in det% at FW=5%':>30s}")
    for label, a, b in EFFECTS:
        diffs = [d[(a, n, s)] - d[(b, n, s)] for (m, n, s) in d
                 if m == a and (b, n, s) in d and not np.isnan(d[(a, n, s)]) and not np.isnan(d[(b, n, s)])]
        if diffs:
            print(f"{label:34s}{np.mean(diffs):+14.1f} points  ({len(diffs)} runs)")


def report(rs, title):
    print(f"\n================ {title} ================")
    print("A) validation-selected parameter = HEADLINE numbers (param = epsilon for boxes, quantile q for ocsvm; reached = runs that hit the 5% target)")
    print_table([r for r in rs if r["selected"]])
    print("\nB) matched false warnings: detection interpolated at test FW = 3 / 5 / 8 % (exploratory ranking only; (k) = runs reaching that FW)")
    print_matched(rs)
    print("\nC) effect at matched FW = 5% (positive = first variant is better)")
    print_effects(rs)


def save():
    with open(OUT_FILE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)


def dead_units(Vd, y, n):
    """Share of constant units inside a class on the raw fit data (zero-width sides)."""
    return " ".join(f"{l}:{100 * np.mean([(Vd[l][y == c].std(0) < 1e-6).mean() for c in range(n) if (y == c).any()]):.1f}%" for l in HIDDEN)


# =====================================================================
# EXPERIMENT
# =====================================================================
print(f"PRESET = {PRESET} | seeds = {SEEDS} | n = {N_RANGE} | {len(V)} variants")
print(f"outliers {100 * OUTLIER_FRAC:.0f}% | floor {FLOOR_FRAC} | percentile {Q_LOW}% | pca {PCA_VAR} | kpca k {KPCA_K} | RFF scale {RFF_SCALE} | "
      f"selection: strictest param with validation FW <= {100 * TARGET_FW:.0f}%")
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
    pred = V_run[-1].argmax(1)                       # always from the RAW outputs
    correct = pred == y_te
    print(f"\nn={n}: accuracy on known digits = {correct[te].mean():.4f}")

    for seed in SEEDS:
        is_val = np.zeros(len(y_ok), dtype=bool)
        is_val[np.random.default_rng(seed).permutation(len(y_ok))[:int(VAL_FRAC * len(y_ok))]] = True
        V_fit, y_fit = take(V_ok, ~is_val), y_ok[~is_val]
        V_val, y_val = take(V_ok, is_val), y_ok[is_val]

        keep = remove_outliers(V_fit, y_fit, n)      # fit part only
        V_cln, y_cln = take(V_fit, keep), y_fit[keep]
        n_removed = int((~keep).sum())
        print(f"   n={n} seed={seed}: removed {n_removed} of {len(keep)} fit points | constant units in a class: {dead_units(V_fit, y_fit, n)}")
        CLUSTERS.clear()

        for name, v in V.items():
            Vf, yf = (V_cln, y_cln) if v["clean"] else (V_fit, y_fit)
            Vv, Vt = V_val, V_run
            t0 = time.perf_counter()
            if v["scaled"]:
                Vf, (Vv, Vt) = scale_apply(Vf, [Vv, Vt])
            if v["tf"] == "rff":
                Vf, (Vv, Vt) = rff_apply(Vf, [Vv, Vt], v["mult"], seed)
            elif v["tf"] == "kpca":
                Vf, (Vv, Vt) = kpca_apply(Vf, [Vv, Vt], v["gmul"], v["k"], seed)
            map_s = time.perf_counter() - t0
            ckey = (v["clean"], v["scaled"], v["tf"], v.get("mult", 0), v.get("gmul", 0), v.get("k", 0))

            seed_all(seed)
            if v["shape"] == "ocsvm":
                mon, grid, ptype = SVMMonitor(v["gmul"], ckey), Q_GRID, "q"
            else:
                mon, grid, ptype = BoxMonitor(v["shape"], ckey), EPSILONS, "epsilon"

            t0 = time.perf_counter()
            mon.fit(Vf, yf, n)
            fit_s = time.perf_counter() - t0         # not comparable between variants (k-means is cached)

            ncl = mon.n_clusters()
            base = dict(seed=seed, n_classes=n, method=name, shape=v["shape"], clean=v["clean"], scaled=v["scaled"],
                        transform=v["tf"], n_clusters=ncl,
                        pts_per_cluster=round(len(LAYERS) * len(yf) / ncl, 1) if ncl else 0,
                        n_support=mon.n_support(), n_fit=len(yf), n_removed=n_removed if v["clean"] else 0,
                        map_s=round(map_s, 2), fit_s=round(fit_s, 2), param_type=ptype)

            mine, chosen = [], None
            for p in grid:
                val_fw = 1 - mon.accepts("val", Vv, y_val, p).mean()
                if chosen is None and val_fw <= TARGET_FW:
                    chosen = p                       # selection uses validation only
                acc = mon.accepts("test", Vt, pred, p)
                row = dict(base, param=p, val_fw_pct=round(100 * val_fw, 2))
                row.update(stats(correct, acc, y_te, n))
                mine.append(row)
            reached = chosen is not None
            if not reached:
                chosen = grid[-1]                    # loosest value
            for r in mine:
                r["selected"] = (r["param"] == chosen)
                r["target_reached"] = reached
            rows.extend(mine)

            s = next(r for r in mine if r["selected"])
            det, fw = test_rates(s)
            print(f"   n={n} seed={seed} {name:30s} {ptype}={chosen:<6} valFW={s['val_fw_pct']:4.1f}% | "
                  f"test det {det:5.1f}%  FW {fw:4.1f}% | map {map_s:.0f}s fit {fit_s:.0f}s"
                  + ("" if reached else "  (target FW not reached)"))

    report([r for r in rows if r["n_classes"] == n], f"n={n}")
    save()                                           # saved after every n

report(rows, "OVERALL (all n, all seeds)")
print("saved", OUT_FILE)