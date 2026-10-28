import csv
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets

from model import MNISTCNY19
from monitor import MultiLayerMonitor, BooleanMonitor


SEED, EPOCHS, BATCH = 0, 10, 128
THRESHOLD, EPSILON = 0.07, 0.0
LAYERS = [-1, -2, -3, -4]
ALPHAS = [0.1, 0.01]
TOTAL_CLASSES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def load():
    tr = datasets.MNIST("./data", train=True, download=True)
    te = datasets.MNIST("./data", train=False, download=True)
    f = lambda d: (d.data.float().unsqueeze(1) - 127.5) / 127.5      # [-1, 1] like the original
    return f(tr), tr.targets.numpy(), f(te), te.targets.numpy()


def train(x, y, n_classes):
    model = MNISTCNY19(n_classes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-7)
    loss_fn = nn.CrossEntropyLoss()
    yt = torch.as_tensor(y, dtype=torch.long)
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(x))
        for i in range(0, len(x), BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            loss_fn(model(x[b].to(DEVICE)), yt[b].to(DEVICE)).backward()
            opt.step()
    return model


@torch.no_grad()
def extract(model, x, bs=2048):
    model.eval()
    vals = {l: [] for l in LAYERS}
    for i in range(0, len(x), bs):
        _, layers = model(x[i:i + bs].to(DEVICE), return_layers=True)
        for l in LAYERS:
            vals[l].append(layers[l].cpu().numpy())
    return {l: np.concatenate(v) for l, v in vals.items()}


def take(vals, mask):
    return {l: v[mask] for l, v in vals.items()}


def stats(correct, accepted, gt_run, n_classes):
    nov = gt_run >= n_classes
    return dict(
        tn=int((correct & accepted).sum()), fn=int((~correct & accepted).sum()),
        tp=int((~correct & ~accepted).sum()), fp=int((correct & ~accepted).sum()),
        nov_det=int((nov & ~accepted).sum()), nov_undet=int((nov & accepted).sum()))


def main():
    seed_all(SEED)
    x_tr, y_tr, x_te, y_te = load()
    rows = []
    for n in range(2, TOTAL_CLASSES):                  # 2..9 known classes
        print(f"\n=== {n} known classes ===")
        tr, te = y_tr < n, y_te < n
        model = train(x_tr[tr], y_tr[tr], n)

        V_tr = extract(model, x_tr[tr]); gt_tr = y_tr[tr]
        ok = V_tr[-1].argmax(1) == gt_tr               # learn only from correct classifications
        V_te = extract(model, x_te[te]);  gt_te = y_te[te]    # monitor-test: known classes only
        V_run = extract(model, x_te)                          # run: all 10 classes
        probs = V_run[-1]
        pred = probs.argmax(1)
        correct = pred == y_te

        V_fit, y_fit = take(V_tr, ok), gt_tr[ok]
        V_comb = {l: np.concatenate([V_fit[l], V_te[l]]) for l in LAYERS}
        y_comb = np.concatenate([y_fit, gt_te])

        kw = dict(threshold=THRESHOLD, epsilon=EPSILON)
        accepted = {
            "monitor1": MultiLayerMonitor(LAYERS, **kw).fit(V_fit, y_fit, n).accepts(V_run, pred),
            "monitor2": MultiLayerMonitor(LAYERS, **kw).fit(V_comb, y_comb, n).accepts(V_run, pred),
            "monitor3": BooleanMonitor().fit(V_fit[-2], y_fit, n).accepts(V_run[-2], pred),
        }
        b = n / (n - 1)
        conf = b * (1 - probs.max(1))                  # alpha thresholding
        for a in ALPHAS:
            accepted[f"at{int(a * 100)}"] = conf <= a

        for name, acc in accepted.items():
            r = dict(n_classes=n, method=name, **stats(correct, acc, y_te, n))
            rows.append(r)
            print(r)

    with open("results_baseline.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()