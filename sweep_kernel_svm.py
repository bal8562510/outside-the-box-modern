import csv
import os
import time
import numpy as np
from sklearn.svm import OneClassSVM

from model import MNISTCNY19
from compare_with_baseline import load_keras_weights
from run_baseline import load, extract, take, stats, LAYERS, DEVICE, seed_all
from monitor_rotated import MultiLayerMonitor, RotatedBox, LayerMonitor, Box

# =====================================================================
# This file only RUNS: "rotated" and "ocsvm" (kernel one-class SVM).
# Box / adaptive results are NOT recomputed: they are read from the old CSV
# and shown in the same tables. Same split, seeds, epsilon grid, selection rule.
# =====================================================================
MODELS_DIR = r"C:\work\original\models"
THRESHOLD = 0.07                       # same as all earlier experiments
FIT_EPS = 0.0                          # same as before

PRESET = "quick_test"
#   "quick_test" : seed 0, n=3 (check for typos and see the run time first)
#   "final"      : seeds 0,1,2 and n = 2..9, same as the earlier final run

PRESETS = {
    "quick_test": dict(seeds=[0], n_range=[3], out="results_ocsvm_quick_test.csv"),
    "final":      dict(seeds=[0, 1, 2], n_range=list(range(2, 10)), out="results_ocsvm_final.csv"),
}
cfg = PRESETS[PRESET]
SEEDS, N_RANGE, OUT_FILE = cfg["seeds"], cfg["n_range"], cfg["out"]

RUN = ["rotated", "ocsvm"]             # remove "rotated" if you do not want to recompute it (it is deterministic)

# Old results to show next to the new ones (not recomputed). Set OLD_CSV = None to skip.
OLD_CSV = "results_final_v2.csv"
OLD_METHODS = ["box", "adaptive_validation"]

# =====================================================================
# SELECTION SETTINGS: identical to the earlier experiments
# =====================================================================
VAL_FRAC = 0.2                         # same 20% held-out known training points
TARGET_FW = 0.05                       # selected epsilon = smallest one whose validation FW <= 5%
EPSILONS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]

# =====================================================================
# KERNEL SVM SETTINGS: fixed, standard values, NOT tuned on anything
# =====================================================================
SVM_NU = 0.1                           # one-class SVM nu
SVM_GAMMA = "scale"                    # RBF width chosen by scikit-learn from the data (1 / (d * variance))
SVM_LAYERS = [-2, -3, -4]              # layer -1 stays a plain Box, as in every other method
ROT_LAYERS = [-2, -3, -4]
ROT = {l: RotatedBox for l in ROT_LAYERS}


class SVMMonitor:
    """One RBF one-class SVM per (layer, class), fitted on the same fit points as the boxes.

    A point is accepted by a layer if its SVM score >= threshold, where
        threshold = s_min - eps * (s_median - s_min)
    and s_min, s_median are the min and median score of the FIT points of that class.
    This mirrors the Box rule: eps = 0 accepts every fit point, larger eps = looser.
    Layer -1 is a plain Box (same k-means clusters as in the other methods).
    The monitor accepts only if EVERY layer accepts, as before.
    """
    def __init__(self, nu, gamma):
        self.nu, self.gamma = nu, gamma
        self.box1 = LayerMonitor(shape_cls=Box, threshold=THRESHOLD, epsilon=0.0)
        self.models = {l: {} for l in SVM_LAYERS}      # layer -> class -> (svm, s_min, spread)
        self.cache = {}                                # SVM scores do not depend on eps: compute once

    def fit(self, V, y, n):
        self.box1.fit(V[-1], y, n)                     # first, so k-means sees the same RNG state as in the other methods
        for l in SVM_LAYERS:
            for c in range(n):
                P = V[l][y == c]
                if len(P) == 0:
                    continue
                svm = OneClassSVM(kernel="rbf", gamma=self.gamma, nu=self.nu).fit(P)
                s = svm.decision_function(P)
                self.models[l][c] = (svm, s.min(), np.median(s) - s.min())
        return self

    def n_support(self):
        return sum(len(m[0].support_) for d in self.models.values() for m in d.values())

    def _scores(self, key, l, V, preds):
        if (key, l) not in self.cache:
            score = np.full(len(V), -np.inf)           # a class without a model is always rejected
            smin, spread = np.zeros(len(V)), np.zeros(len(V))
            for c, (svm, lo, sp) in self.models[l].items():
                idx = preds == c
                if idx.any():
                    score[idx] = svm.decision_function(V[idx])
                    smin[idx], spread[idx] = lo, sp
            self.cache[(key, l)] = (score, smin, spread)
        return self.cache[(key, l)]

    def accepts(self, key, V, preds, eps):
        self.box1.epsilon = eps
        ok = np.array(self.box1.accepts(V[-1], preds), dtype=bool)
        for l in SVM_LAYERS:
            score, smin, spread = self._scores(key, l, V[l], preds)
            ok &= score >= smin - eps * spread
        return ok


def test_rates(r):
    return (100 * r["nov_det"] / (r["nov_det"] + r["nov_undet"]),
            100 * r["fp"] / (r["fp"] + r["tn"]))


def load_old(path, methods):
    """Saved results of the earlier runs: validation-selected rows only, same n and seeds."""
    if not path or not os.path.exists(path):
        print(f"(old results not found: {path}) -> showing only the new methods")
        return []
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["method"] in methods and r["selected"] == "True":
                for k in ("tn", "fp", "nov_det", "nov_undet", "n_classes", "seed"):
                    r[k] = int(r[k])
                for k in ("epsilon", "val_fw_pct", "n_clusters", "fit_s"):
                    r[k] = float(r[k])
                if r["n_classes"] in N_RANGE and r["seed"] in SEEDS:
                    out.append(r)
    return out


def print_table(rs):
    """Validation-selected epsilon only: one line per method (mean over the rows given)."""
    print(f"{'method':22s}{'eps':>6s}{'valFW%':>8s}{'det%':>8s}{'FW%':>8s}{'clusters':>10s}{'fit_s':>8s}")
    for label in dict.fromkeys(r["method"] for r in rs):
        g = [r for r in rs if r["method"] == label]
        t = np.array([test_rates(r) for r in g])
        print(f"{label:22s}{np.mean([r['epsilon'] for r in g]):6.2f}{np.mean([r['val_fw_pct'] for r in g]):8.1f}"
              f"{t[:, 0].mean():8.1f}{t[:, 1].mean():8.1f}"
              f"{np.mean([r['n_clusters'] for r in g]):10.1f}{np.mean([r['fit_s'] for r in g]):8.1f}")


def print_grid(rs):
    """All epsilons (new methods): cell = test detected% / test false-warning%."""
    print(f"{'method':22s}" + "".join(f"{'eps=' + str(e):>13s}" for e in EPSILONS))
    for label in dict.fromkeys(r["method"] for r in rs):
        line = f"{label:22s}"
        for e in EPSILONS:
            g = [r for r in rs if r["method"] == label and r["epsilon"] == e]
            t = np.array([test_rates(r) for r in g])
            line += f"{t[:, 0].mean():6.1f}/{t[:, 1].mean():5.1f} "
        print(line)


def report(new_rows, old_rows, title):
    print(f"\n--- {title}: epsilon selected on validation (mean over seeds; old methods are read from {OLD_CSV}) ---")
    print_table(old_rows + [r for r in new_rows if r["selected"]])
    print(f"\n--- {title}: ALL epsilons, new methods, cell = test det% / test FW% (do not pick the best from here) ---")
    print_grid(new_rows)


def save():
    with open(OUT_FILE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)


# =====================================================================
# EXPERIMENT
# =====================================================================
print(f"PRESET = {PRESET} | methods run now = {RUN} | seeds = {SEEDS} | n = {N_RANGE}")
print(f"OCSVM: RBF kernel, gamma={SVM_GAMMA}, nu={SVM_NU} (fixed, not tuned) on layers {SVM_LAYERS}; layer -1 = Box")
print(f"selected epsilon = smallest one with validation FW <= {100 * TARGET_FW:.0f}% (known held-out training points)")
x_tr, y_tr, x_te, y_te = load()
old = load_old(OLD_CSV, OLD_METHODS)
rows = []

for n in N_RANGE:
    model = MNISTCNY19(n)
    load_keras_weights(model, rf"{MODELS_DIR}\CNY19id1_MNIST_0-{n-1}.h5")
    model.to(DEVICE).eval()

    tr, te = y_tr < n, y_te < n
    V_tr = extract(model, x_tr[tr]); gt_tr = y_tr[tr]
    ok = V_tr[-1].argmax(1) == gt_tr
    V_ok, y_ok = take(V_tr, ok), gt_tr[ok]           # known, correctly classified training points
    V_run = extract(model, x_te)
    pred = V_run[-1].argmax(1)                       # always from the RAW outputs
    correct = pred == y_te
    print(f"\nn={n}: accuracy on known digits = {correct[te].mean():.4f}")

    for seed in SEEDS:
        # identical split to the earlier runs (same seed -> same permutation)
        is_val = np.zeros(len(y_ok), dtype=bool)
        is_val[np.random.default_rng(seed).permutation(len(y_ok))[:int(VAL_FRAC * len(y_ok))]] = True
        V_fit, y_fit = take(V_ok, ~is_val), y_ok[~is_val]
        V_val, y_val = take(V_ok, is_val), y_ok[is_val]

        for name in RUN:
            seed_all(seed)                           # same k-means seed as in the earlier runs
            if name == "ocsvm":
                mon = SVMMonitor(SVM_NU, SVM_GAMMA)
                def accepts(key, V, preds, eps, mon=mon):
                    return mon.accepts(key, V, preds, eps)
            else:
                mon = MultiLayerMonitor(LAYERS, shape_by_layer=ROT, threshold=THRESHOLD, epsilon=FIT_EPS)
                def accepts(key, V, preds, eps, mon=mon):
                    for m in mon.mons.values():
                        m.epsilon = eps
                    return mon.accepts(V, preds)

            t0 = time.perf_counter()
            mon.fit(V_fit, y_fit, n)
            fit_s = time.perf_counter() - t0

            if name == "ocsvm":
                n_clusters, n_rotated, n_support = 0, 0, mon.n_support()
            else:
                shapes = [s for m in mon.mons.values() for lst in m.shapes.values() for s in lst]
                n_clusters, n_rotated, n_support = len(shapes), sum(isinstance(s, RotatedBox) for s in shapes), 0
            base = dict(seed=seed, n_classes=n, method=name, n_clusters=n_clusters, n_rotated=n_rotated,
                        fit_s=round(fit_s, 3), rff_s=0.0, rff_sigma_scale=0, rff_dims="", n_support=n_support)

            mine, chosen = [], None
            for eps in EPSILONS:
                val_fw = 1 - accepts("val", V_val, y_val, eps).mean()
                if chosen is None and val_fw <= TARGET_FW:
                    chosen = eps                     # selection uses validation only
                acc = accepts("test", V_run, pred, eps)
                row = dict(base, epsilon=eps, val_fw_pct=round(100 * val_fw, 2))
                row.update(stats(correct, acc, y_te, n))
                mine.append(row)
            reached = chosen is not None
            if not reached:
                chosen = EPSILONS[-1]
            for r in mine:
                r["selected"] = (r["epsilon"] == chosen)
                r["target_reached"] = reached
            rows.extend(mine)

            s = next(r for r in mine if r["selected"])
            det, fw = test_rates(s)
            print(f"   n={n} seed={seed} {name:10s} selected eps={chosen:<4} valFW={s['val_fw_pct']:4.1f}% | "
                  f"test det {det:5.1f}%  FW {fw:4.1f}% | fit {fit_s:.0f}s"
                  + ("" if reached else "  (target FW not reached)"))

    report([r for r in rows if r["n_classes"] == n], [r for r in old if r["n_classes"] == n], f"n={n}")
    save()                                           # saved after every n

report(rows, old, "OVERALL (all n, all seeds)")
print("saved", OUT_FILE)