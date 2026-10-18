import csv
import numpy as np
from model import MNISTCNY19
from compare_with_baseline import load_keras_weights
from run_baseline import load, extract, take, stats, LAYERS, DEVICE, seed_all
from monitor import MultiLayerMonitor, BooleanMonitor

MODELS_DIR = r"C:\work\original\models"
SEEDS = [0, 1, 2]          # only k-means randomness differs between seeds
ALPHAS = [0.1, 0.01]

x_tr, y_tr, x_te, y_te = load()
rows = []

for n in range(2, 10):
    model = MNISTCNY19(n)
    load_keras_weights(model, rf"{MODELS_DIR}\CNY19id1_MNIST_0-{n-1}.h5")
    model.to(DEVICE).eval()

    tr, te = y_tr < n, y_te < n
    V_tr = extract(model, x_tr[tr]); gt_tr = y_tr[tr]
    V_run = extract(model, x_te)
    probs = V_run[-1]
    pred = probs.argmax(1)
    correct = pred == y_te
    print(f"n={n}: accuracy on known digits = {correct[te].mean():.4f}")

    ok = V_tr[-1].argmax(1) == gt_tr
    V_fit, y_fit = take(V_tr, ok), gt_tr[ok]

    # box monitor (monitor1): several seeds
    for seed in SEEDS:
        seed_all(seed)
        mon = MultiLayerMonitor(LAYERS, threshold=0.07, epsilon=0.0).fit(V_fit, y_fit, n)
        acc = mon.accepts(V_run, pred)
        rows.append(dict(seed=seed, n_classes=n, method="monitor1",
                         **stats(correct, acc, y_te, n)))

    # deterministic methods: one run each
    acc = BooleanMonitor().fit(V_fit[-2], y_fit, n).accepts(V_run[-2], pred)
    rows.append(dict(seed=-1, n_classes=n, method="monitor3",
                     **stats(correct, acc, y_te, n)))
    b = n / (n - 1)
    for a in ALPHAS:
        acc = (b * (1 - probs.max(1))) <= a
        rows.append(dict(seed=-1, n_classes=n, method=f"at{int(a*100)}",
                         **stats(correct, acc, y_te, n)))

with open("results_pretrained.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)

print("\nmonitor1 summary (mean +- std over seeds)")
print("n | novelties detected % | false warnings %")
for n in range(2, 10):
    r = [x for x in rows if x["method"] == "monitor1" and x["n_classes"] == n]
    nd = [100 * x["nov_det"] / (x["nov_det"] + x["nov_undet"]) for x in r]
    fw = [100 * x["fp"] / (x["fp"] + x["tn"]) for x in r]
    print(f"{n} | {np.mean(nd):.1f} +- {np.std(nd):.1f} | {np.mean(fw):.1f} +- {np.std(fw):.1f}")