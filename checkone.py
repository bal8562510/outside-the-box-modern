from model import MNISTCNY19
from compare_with_baseline import load_keras_weights, WEIGHTS_PATH
from run_baseline import load, extract, take, stats, LAYERS, DEVICE, seed_all
from monitor import MultiLayerMonitor

seed_all(0)
n = 3                                    # digits 0,1,2 are known
x_tr, y_tr, x_te, y_te = load()
tr = y_tr < n

# original network (pretrained)
model = MNISTCNY19(n)
load_keras_weights(model, WEIGHTS_PATH)
model.to(DEVICE).eval()

# build boxes from correctly classified training samples
V_tr = extract(model, x_tr[tr]); gt_tr = y_tr[tr]
ok = V_tr[-1].argmax(1) == gt_tr
mon = MultiLayerMonitor(LAYERS, threshold=0.07, epsilon=0.0)
mon.fit(take(V_tr, ok), gt_tr[ok], n)

# test on ALL digits (7-9 etc. are the novelties)
V_run = extract(model, x_te)
pred = V_run[-1].argmax(1)
accepted = mon.accepts(V_run, pred)
s = stats(pred == y_te, accepted, y_te, n)

print(s)
print("novelties detected: %.1f %%" % (100 * s["nov_det"] / (s["nov_det"] + s["nov_undet"])))
print("false warnings:     %.1f %%" % (100 * s["fp"] / (s["fp"] + s["tn"])))