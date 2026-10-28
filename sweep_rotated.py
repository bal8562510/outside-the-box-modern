import csv
import numpy as np

from model import MNISTCNY19
from compare_with_baseline import load_keras_weights
from run_baseline import load, extract, take, stats, LAYERS, DEVICE, seed_all
from monitor_rotated import MultiLayerMonitor, RotatedBox

# ---------------------------------------------------------------------
# Same pipeline as sweep_pretrained.py. Only the SHAPE changes.
#   "box"     = baseline (axis-aligned box on all 4 layers) -> control
#   "rotated" = PCA-rotated box on layers -2, -3, -4; layer -1 (softmax,
#               values on a simplex) stays an axis-aligned box
# Both use the same seed before fitting, so k-means gives the same clusters.
# ---------------------------------------------------------------------
MODELS_DIR = r"C:\work\original\models"
SEEDS = [0, 1, 2]            # start with [0] if you want a quick first run
EPSILONS = [0.0]             # baseline value; add e.g. 0.05, 0.1 later
THRESHOLD = 0.07

SHAPES = {
    "box": None,
    "rotated": {-2: RotatedBox, -3: RotatedBox, -4: RotatedBox},
}

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

    ok = V_tr[-1].argmax(1) == gt_tr          # learn only from correct classifications
    V_fit, y_fit = take(V_tr, ok), gt_tr[ok]

    for eps in EPSILONS:
        for seed in SEEDS:
            for name, shape_by_layer in SHAPES.items():
                seed_all(seed)                 # identical clustering for every shape
                mon = MultiLayerMonitor(LAYERS, shape_by_layer=shape_by_layer,
                                        threshold=THRESHOLD, epsilon=eps)
                mon.fit(V_fit, y_fit, n)
                acc = mon.accepts(V_run, pred)
                rows.append(dict(seed=seed, n_classes=n, method=name, epsilon=eps,
                                 **stats(correct, acc, y_te, n)))
        print(f"   done eps={eps}")

with open("results_rotated.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)


def rates(rs):
    nd = [100 * r["nov_det"] / (r["nov_det"] + r["nov_undet"]) for r in rs]
    fw = [100 * r["fp"] / (r["fp"] + r["tn"]) for r in rs]
    return np.mean(nd), np.std(nd), np.mean(fw), np.std(fw)


for eps in EPSILONS:
    print(f"\nepsilon = {eps}   (mean +- std over seeds)")
    print("n | box: detected % / false warn % | rotated: detected % / false warn %")
    for n in range(2, 10):
        out = []
        for name in SHAPES:
            rs = [r for r in rows if r["method"] == name and r["n_classes"] == n
                  and r["epsilon"] == eps]
            a, b, c, d = rates(rs)
            out.append(f"{a:.1f}+-{b:.1f} / {c:.1f}+-{d:.1f}")
        print(f"{n} | " + " | ".join(out))