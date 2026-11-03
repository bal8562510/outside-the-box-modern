import csv
import time
import numpy as np

from model import MNISTCNY19
from compare_with_baseline import load_keras_weights
from run_baseline import load, extract, take, stats, LAYERS, DEVICE, seed_all
from monitor_rotated import MultiLayerMonitor, RotatedBox
from rff import RFF                                                    # RFF (rff.py unchanged)

MODELS_DIR = r"C:\work\original\models"
THRESHOLD = 0.07                       # same as all earlier experiments

# =====================================================================
# 1. CHOOSE THE EXPERIMENT HERE (change only this line)
# =====================================================================
PRESET = "epsilon_rff"
#   "quick_test" : 1 seed, n=3 only, all 4 methods, 4 epsilons (a few minutes; check for typos)
#   "original"   : your first experiment: box vs rotated, eps 0.0, 3 seeds
#   "adaptive"   : box, rotated, adaptive_volume, adaptive_validation, eps 0.0
#   "epsilon"    : box, rotated, adaptive_validation, eps 0.0/0.05/0.1/0.2
#   "quick_test_rff" : like quick_test but with the RFF methods added      # RFF
#   "epsilon_rff"    : "epsilon" + the RFF methods (current plan)          # RFF

PRESETS = {
    "quick_test": dict(seeds=[0], n_range=range(3, 4),
                       run=["box", "rotated", "adaptive_volume", "adaptive_validation"],
                       epsilons=[0.0, 0.05, 0.1, 0.2],
                       out="results_quick_test.csv"),
    "original":   dict(seeds=[0, 1, 2], n_range=range(2, 10),
                       run=["box", "rotated"],
                       epsilons=[0.0],
                       out="results_rotated.csv"),
    "adaptive":   dict(seeds=[0, 1, 2], n_range=range(2, 10),
                       run=["box", "rotated", "adaptive_volume", "adaptive_validation"],
                       epsilons=[0.0],
                       out="results_adaptive.csv"),
    "epsilon":    dict(seeds=[0, 1, 2], n_range=range(2, 10),
                       run=["box", "rotated", "adaptive_validation"],
                       epsilons=[0.0, 0.05, 0.1, 0.2],
                       out="results_epsilon_with_adaptive.csv"),
    # RFF: new presets, the old ones above are untouched
    "quick_test_rff": dict(seeds=[0], n_range=range(3, 4),
                           run=["box", "rotated", "adaptive_validation", "rff_d1", "rff_d4"],
                           epsilons=[0.0, 0.05, 0.1, 0.2],
                           out="results_quick_test_rff.csv"),
    "epsilon_rff":    dict(seeds=[0, 1, 2], n_range=range(2, 10),
                           run=["box", "rotated", "adaptive_validation", "rff_d1", "rff_d4"],
                           epsilons=[0.0, 0.05, 0.1, 0.2],
                           out="results_epsilon_with_rff.csv"),
}

cfg = PRESETS[PRESET]
SEEDS, N_RANGE = cfg["seeds"], cfg["n_range"]
RUN, EPSILONS, OUT_FILE = cfg["run"], cfg["epsilons"], cfg["out"]

# Epsilon used while FITTING. Only matters for adaptive_validation
# (it uses epsilon when checking the hidden points). Box and rotated ignore it.
FIT_EPS = 0.0

# =====================================================================
# 2. SHAPE POLICIES (layer -1 stays a plain Box in every method)
# =====================================================================
ROT_LAYERS = [-2, -3, -4]
SHAPES = {
    "box": None,
    "rotated": {l: RotatedBox for l in ROT_LAYERS},
    "adaptive_volume":     {l: "adaptive_volume" for l in ROT_LAYERS},
    "adaptive_validation": {l: "adaptive_validation" for l in ROT_LAYERS},
    "rff_d1": None,                    # RFF: plain Box on RFF-transformed data (D = 1 x layer dim)
    "rff_d4": None,                    # RFF: plain Box on RFF-transformed data (D = 4 x layer dim)
}
# To add a new method later: add an entry to SHAPES, then add its name to "run" in a preset.

# =====================================================================
# 2b. RFF SETTINGS (only used by methods whose name starts with "rff")
# =====================================================================
RFF_LAYERS = [-2, -3, -4]              # RFF: layers that get the map (layer -1 stays raw Box)
RFF_FACTORS = {"rff_d1": 1, "rff_d4": 4}   # RFF: D = factor * layer dimension (d1 = control)
RFF_D_MAX = 1024                       # RFF: safety cap on D
RFF_SIGMA_SCALE = 2.0                  # RFF: same value as the earlier run


def apply_rff(V_fit, V_run, seed, factor):                             # RFF
    """One RFF map per layer, fitted on the FIT data only, applied to fit AND run data.
    Returns the transformed dicts and the actual D used per layer."""
    Vf, Vr, dims = dict(V_fit), dict(V_run), {}
    for l in RFF_LAYERS:
        D = int(min(factor * V_fit[l].shape[1], RFF_D_MAX))
        f = RFF(D, RFF_SIGMA_SCALE, seed=seed * 100 + abs(l)).fit(V_fit[l])
        Vf[l] = f.transform(V_fit[l])
        Vr[l] = f.transform(V_run[l])
        dims[l] = D
    return Vf, Vr, dims


# =====================================================================
# 3. EXPERIMENT
# =====================================================================
print(f"PRESET = {PRESET} | methods = {RUN} | epsilons = {EPSILONS} | seeds = {SEEDS}")
x_tr, y_tr, x_te, y_te = load()
rows = []

for n in N_RANGE:
    model = MNISTCNY19(n)
    load_keras_weights(model, rf"{MODELS_DIR}\CNY19id1_MNIST_0-{n-1}.h5")
    model.to(DEVICE).eval()

    tr, te = y_tr < n, y_te < n
    V_tr = extract(model, x_tr[tr]); gt_tr = y_tr[tr]
    V_run = extract(model, x_te)
    pred = V_run[-1].argmax(1)               # always from the RAW outputs
    correct = pred == y_te
    print(f"n={n}: accuracy on known digits = {correct[te].mean():.4f}")

    ok = V_tr[-1].argmax(1) == gt_tr
    V_fit, y_fit = take(V_tr, ok), gt_tr[ok]

    for seed in SEEDS:
        for name in RUN:
            seed_all(seed)                           # identical k-means seed for every method
            mon = MultiLayerMonitor(LAYERS, shape_by_layer=SHAPES[name],
                                    threshold=THRESHOLD, epsilon=FIT_EPS)

            # RFF: transform fit AND run data with the same map (other methods use raw data)
            t0 = time.perf_counter()
            dims = {}
            if name.startswith("rff"):
                Vf, Vr, dims = apply_rff(V_fit, V_run, seed, RFF_FACTORS[name])
            else:
                Vf, Vr = V_fit, V_run
            rff_s = time.perf_counter() - t0

            t0 = time.perf_counter()
            mon.fit(Vf, y_fit, n)                    # fit ONCE per (n, seed, method)
            fit_s = time.perf_counter() - t0

            all_shapes = [s for m in mon.mons.values()
                          for lst in m.shapes.values() for s in lst]
            n_clusters = len(all_shapes)
            n_rotated = sum(isinstance(s, RotatedBox) for s in all_shapes)

            for eps in EPSILONS:                     # only re-evaluate per epsilon
                for m in mon.mons.values():
                    m.epsilon = eps
                acc = mon.accepts(Vr, pred)
                rows.append(dict(seed=seed, n_classes=n, method=name, epsilon=eps,
                                 **stats(correct, acc, y_te, n),
                                 n_clusters=n_clusters, n_rotated=n_rotated,
                                 fit_s=round(fit_s, 3),
                                 rff_s=round(rff_s, 3),                                   # RFF
                                 rff_sigma_scale=RFF_SIGMA_SCALE if dims else 0,          # RFF
                                 rff_dims=";".join(f"{l}:{d}" for l, d in dims.items())))  # RFF
    print(f"   done n={n}")

with open(OUT_FILE, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)
print("saved", OUT_FILE)


# =====================================================================
# 4. SUMMARY TABLES
# =====================================================================
def rates(rs):
    nd = [100 * r["nov_det"] / (r["nov_det"] + r["nov_undet"]) for r in rs]
    fw = [100 * r["fp"] / (r["fp"] + r["tn"]) for r in rs]
    return np.mean(nd), np.std(nd), np.mean(fw), np.std(fw)


for eps in EPSILONS:
    print(f"\nepsilon = {eps}   detected % / false warn %  (mean +- std over seeds)")
    print("n | " + " | ".join(RUN))
    for n in N_RANGE:
        out = []
        for name in RUN:
            rs = [r for r in rows if r["method"] == name and r["n_classes"] == n
                  and r["epsilon"] == eps]
            a, b, c, d = rates(rs)
            out.append(f"{a:.1f}+-{b:.1f} / {c:.1f}+-{d:.1f}")
        print(f"{n} | " + " | ".join(out))

# Average over all n and seeds: one line per method and epsilon, so the
# detection / false-warning trade-off can be compared at a glance.        # RFF
print("\nAverage over all n and seeds (detected % / false warn %)")
print("method".ljust(22) + " | ".join(f"eps={e}".ljust(16) for e in EPSILONS))
for name in RUN:
    cells = []
    for eps in EPSILONS:
        rs = [r for r in rows if r["method"] == name and r["epsilon"] == eps]
        a, _, c, _ = rates(rs)
        cells.append(f"{a:5.1f} / {c:5.1f}".ljust(16))
    print(name.ljust(22) + " | ".join(cells))

print("\nMean number of clusters, RFF transform time and monitor fit time")
for name in RUN:
    rs = [r for r in rows if r["method"] == name]
    print(f"{name:22s} clusters {np.mean([r['n_clusters'] for r in rs]):7.1f}   "
          f"rff {np.mean([r['rff_s'] for r in rs]):.2f}s   "
          f"fit {np.mean([r['fit_s'] for r in rs]):.2f}s")