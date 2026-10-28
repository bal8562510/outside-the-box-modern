import csv
import time

import numpy as np

from compare_with_baseline import load_keras_weights
from model import MNISTCNY19
from monitor_rotated import MultiLayerMonitor, RotatedBox
from rff import RFF  # RFF
from run_baseline import DEVICE, LAYERS, extract, load, seed_all, stats, take

MODELS_DIR = r"D:\Projects\Outside-the-Box\outside-the-box-modern\work\original"
SEEDS = [0, 1, 2]
EPSILONS = [0.0]
THRESHOLD = 0.07

ROT_LAYERS = [-2, -3, -4]                 # layer -1 stays Box

RFF_LAYERS = [-2, -3, -4]                 # RFF: layers that get the map (layer -1 stays raw Box)
RFF_D_FACTOR = 4                          # RFF: D = factor * layer dimension
RFF_D_MAX = 1024                          # RFF: safety cap on D (memory / k-means time)
RFF_SIGMA_SCALE = 2.0                     # RFF: larger = lower frequencies = fewer wrap-arounds

SHAPES = {
    "box": None,
    "rotated": {l: RotatedBox for l in ROT_LAYERS},
    "adaptive_volume":     {l: "adaptive_volume" for l in ROT_LAYERS},
    "adaptive_validation": {l: "adaptive_validation" for l in ROT_LAYERS},
    "rff": None,                          # RFF: plain Box, but on transformed data
}

# Choose what to run. Old experiment = ["box", "rotated"]
RUN = ["box", "rff"]

# Output file: RFF runs encode their settings in the name so sweeps never overwrite each other
if "rff" in RUN:
    OUT_FILE = f"results_rff_D{RFF_D_FACTOR}_s{RFF_SIGMA_SCALE}.csv"
elif any(r.startswith("adaptive") for r in RUN):
    OUT_FILE = "results_adaptive.csv"
else:
    OUT_FILE = "results_rotated.csv"


def apply_rff(V_fit, V_run, seed):                                    # RFF
    """One RFF map per layer, fitted on the FIT data only, applied to fit AND run data.
    Returns the transformed dicts and the actual D used per layer."""
    Vf, Vr, dims = dict(V_fit), dict(V_run), {}
    for l in RFF_LAYERS:
        D = int(min(RFF_D_FACTOR * V_fit[l].shape[1], RFF_D_MAX))
        f = RFF(D, RFF_SIGMA_SCALE, seed=seed * 100 + abs(l)).fit(V_fit[l])
        Vf[l] = f.transform(V_fit[l])
        Vr[l] = f.transform(V_run[l])
        dims[l] = D
    return Vf, Vr, dims


x_tr, y_tr, x_te, y_te = load()
rows = []

for n in range(2, 10):
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

    # diagnostics: layer sizes and dead (constant) units in the fit data
    print("   layer dims :", {l: V_fit[l].shape[1] for l in LAYERS})
    print("   dead units :", {l: int((V_fit[l].std(0) == 0).sum()) for l in RFF_LAYERS})

    for eps in EPSILONS:
        for seed in SEEDS:
            for name in RUN:
                seed_all(seed)
                mon = MultiLayerMonitor(LAYERS, shape_by_layer=SHAPES[name],
                                        threshold=THRESHOLD, epsilon=eps)

                # RFF transform time is measured separately from monitor fit time
                t0 = time.perf_counter()
                dims = {}
                if name == "rff":
                    Vf, Vr, dims = apply_rff(V_fit, V_run, seed)
                else:
                    Vf, Vr = V_fit, V_run
                rff_s = time.perf_counter() - t0

                t0 = time.perf_counter()
                mon.fit(Vf, y_fit, n)
                fit_s = time.perf_counter() - t0
                acc = mon.accepts(Vr, pred)

                all_shapes = [s for m in mon.mons.values()
                              for lst in m.shapes.values() for s in lst]
                n_clusters = len(all_shapes)
                n_rotated = sum(isinstance(s, RotatedBox) for s in all_shapes)

                rows.append(dict(seed=seed, n_classes=n, method=name, epsilon=eps,
                                 **stats(correct, acc, y_te, n),
                                 n_clusters=n_clusters, n_rotated=n_rotated,
                                 fit_s=round(fit_s, 3), rff_s=round(rff_s, 3),
                                 rff_D_factor=RFF_D_FACTOR if name == "rff" else 0,
                                 rff_sigma_scale=RFF_SIGMA_SCALE if name == "rff" else 0,
                                 rff_dims=";".join(f"{l}:{d}" for l, d in dims.items())))
        print(f"   done eps={eps}")

with open(OUT_FILE, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)
print("saved", OUT_FILE)


def rates(rs):
    nd = [100 * r["nov_det"] / (r["nov_det"] + r["nov_undet"]) for r in rs]
    fw = [100 * r["fp"] / (r["fp"] + r["tn"]) for r in rs]
    return np.mean(nd), np.std(nd), np.mean(fw), np.std(fw)


for eps in EPSILONS:
    print(f"\nepsilon = {eps}   (mean +- std over seeds)   novelty detected % / false warnings %")
    print("n | " + " | ".join(RUN))
    for n in range(2, 10):
        out = []
        for name in RUN:
            rs = [r for r in rows if r["method"] == name and r["n_classes"] == n
                  and r["epsilon"] == eps]
            a, b, c, d = rates(rs)
            out.append(f"{a:.1f}+-{b:.1f} / {c:.1f}+-{d:.1f}")
        print(f"{n} | " + " | ".join(out))

    print("\nMean number of clusters, RFF transform time and monitor fit time")
    for name in RUN:
        rs = [r for r in rows if r["method"] == name and r["epsilon"] == eps]
        print(f"{name:20s} clusters {np.mean([r['n_clusters'] for r in rs]):7.1f}   "
              f"rff {np.mean([r['rff_s'] for r in rs]):.2f}s   "
              f"fit {np.mean([r['fit_s'] for r in rs]):.2f}s")