import csv
import time                                                      # NEW
import numpy as np

from model import MNISTCNY19
from compare_with_baseline import load_keras_weights
from run_baseline import load, extract, take, stats, LAYERS, DEVICE, seed_all
from monitor_rotated import MultiLayerMonitor, RotatedBox

MODELS_DIR =  r"C:\work\original\models"         
SEEDS = [0, 1, 2]
EPSILONS = [0.0]
THRESHOLD = 0.07

ROT_LAYERS = [-2, -3, -4]                                        # NEW (layer -1 stays Box)

SHAPES = {
    "box": None,
    "rotated": {l: RotatedBox for l in ROT_LAYERS},
    "adaptive_volume":     {l: "adaptive_volume" for l in ROT_LAYERS},      # NEW
    "adaptive_validation": {l: "adaptive_validation" for l in ROT_LAYERS},  # NEW
}

# NEW: choose what to run. Old experiment = ["box", "rotated"]

RUN = ["box", "rotated", "adaptive_volume", "adaptive_validation"]

# NEW: old runs keep writing results_rotated.csv, runs with adaptive use a new file
OUT_FILE = ("results_adaptive.csv" if any(r.startswith("adaptive") for r in RUN)
            else "results_rotated.csv")

x_tr, y_tr, x_te, y_te = load()
rows = []

for n in range(2, 10):
    model = MNISTCNY19(n)
    load_keras_weights(model, rf"{MODELS_DIR}\CNY19id1_MNIST_0-{n-1}.h5")
    model.to(DEVICE).eval()

    tr, te = y_tr < n, y_te < n
    V_tr = extract(model, x_tr[tr]); gt_tr = y_tr[tr]
    V_run = extract(model, x_te)
    pred = V_run[-1].argmax(1)
    correct = pred == y_te
    print(f"n={n}: accuracy on known digits = {correct[te].mean():.4f}")

    ok = V_tr[-1].argmax(1) == gt_tr
    V_fit, y_fit = take(V_tr, ok), gt_tr[ok]

    for eps in EPSILONS:
        for seed in SEEDS:
            for name in RUN:                                     # CHANGED: was SHAPES.items()
                seed_all(seed)
                mon = MultiLayerMonitor(LAYERS, shape_by_layer=SHAPES[name],
                                        threshold=THRESHOLD, epsilon=eps)
                t0 = time.perf_counter()                         # NEW
                mon.fit(V_fit, y_fit, n)
                fit_s = time.perf_counter() - t0                 # NEW
                acc = mon.accepts(V_run, pred)

                all_shapes = [s for m in mon.mons.values()       # NEW
                              for lst in m.shapes.values() for s in lst]
                n_clusters = len(all_shapes)                     # NEW
                n_rotated = sum(isinstance(s, RotatedBox) for s in all_shapes)  # NEW

                rows.append(dict(seed=seed, n_classes=n, method=name, epsilon=eps,
                                 **stats(correct, acc, y_te, n),
                                 n_clusters=n_clusters, n_rotated=n_rotated,   # NEW
                                 fit_s=round(fit_s, 3)))                       # NEW
        print(f"   done eps={eps}")

with open(OUT_FILE, "w", newline="") as f:                       # CHANGED: file name
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)
print("saved", OUT_FILE)


def rates(rs):
    nd = [100 * r["nov_det"] / (r["nov_det"] + r["nov_undet"]) for r in rs]
    fw = [100 * r["fp"] / (r["fp"] + r["tn"]) for r in rs]
    return np.mean(nd), np.std(nd), np.mean(fw), np.std(fw)


for eps in EPSILONS:
    print(f"\nepsilon = {eps}   (mean +- std over seeds)")
    print("n | " + " | ".join(RUN))
    for n in range(2, 10):
        out = []
        for name in RUN:                                         # CHANGED
            rs = [r for r in rows if r["method"] == name and r["n_classes"] == n
                  and r["epsilon"] == eps]
            a, b, c, d = rates(rs)
            out.append(f"{a:.1f}+-{b:.1f} / {c:.1f}+-{d:.1f}")
        print(f"{n} | " + " | ".join(out))

    print("\nShare of clusters that use a rotated box, and fit time")   # NEW
    for name in RUN:
        rs = [r for r in rows if r["method"] == name and r["epsilon"] == eps]
        frac = 100 * sum(r["n_rotated"] for r in rs) / sum(r["n_clusters"] for r in rs)
        print(f"{name:20s} rotated clusters {frac:5.1f}%   "
              f"mean fit time {np.mean([r['fit_s'] for r in rs]):.2f}s")