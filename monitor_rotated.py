import numpy as np
from sklearn.cluster import KMeans


# =====================================================================
# This file is a copy of monitor.py (baseline-v1) with two additions:
#   1. the class RotatedBox (new shape, placed right after Box)
#   2. MultiLayerMonitor accepts an optional 'shape_by_layer' dict so
#      that different layers can use different shapes.
# Everything else is identical to monitor.py.
# =====================================================================


def cluster_refinement(X, threshold):
    """Port of utils/Clustering.py: add clusters until inertia improves < threshold."""
    n = 1
    clustered = KMeans(n, n_init=10).fit(X)
    inertias = [clustered.inertia_]
    while len(X) > n:
        new = KMeans(n + 1, n_init=10).fit(X)
        inertias.append(new.inertia_)
        if 1 - inertias[-1] / inertias[-2] < threshold:
            break                        # keep the previous clustering
        clustered = new
        n += 1
    return clustered


class Box:
    """epsilon is relative: each side grows by (high-low)/2 * eps."""
    def __init__(self, pts):
        self.low, self.high = pts.min(0), pts.max(0)

    def contains(self, X, eps=0.0):
        pad = (self.high - self.low) / 2.0 * eps
        return np.all((X >= self.low - pad) & (X <= self.high + pad), axis=1)


class RotatedBox:
    """PCA-rotated box (NEW).

    1. Centre the cluster at its mean.
    2. Rotate into the principal axes of the cluster (eigenvectors of the
       covariance matrix; the FULL d x d basis is kept, so directions with
       zero variance become zero-width sides, exactly like constant
       dimensions in the axis-aligned Box).
    3. Take the min/max box in the rotated coordinates.

    epsilon is relative, as in Box: each side grows by (high-low)/2 * eps,
    but measured in the rotated coordinates.
    TOL is a tiny absolute tolerance (1e-6) that absorbs float round-off
    after the rotation; the axis-aligned Box has none.
    """
    TOL = 1e-6

    def __init__(self, pts):
        pts = np.asarray(pts, dtype=np.float64)
        self.mu = pts.mean(0)
        centered = pts - self.mu
        cov = centered.T @ centered / len(pts)
        _, vecs = np.linalg.eigh(cov)            # columns = orthonormal axes
        self.R = vecs
        z = centered @ self.R
        self.low, self.high = z.min(0), z.max(0)

    def contains(self, X, eps=0.0):
        z = (np.asarray(X, dtype=np.float64) - self.mu) @ self.R
        pad = (self.high - self.low) / 2.0 * eps + self.TOL
        return np.all((z >= self.low - pad) & (z <= self.high + pad), axis=1)


class LayerMonitor:
    """One set of shapes per class for ONE layer (one shape per k-means cluster)."""
    def __init__(self, shape_cls=Box, threshold=0.07, epsilon=0.0):
        self.shape_cls, self.threshold, self.epsilon = shape_cls, threshold, epsilon
        self.shapes = {}

    def fit(self, V, y, n_classes):
        for c in range(n_classes):
            pts = V[y == c]
            if len(pts) == 0:
                self.shapes[c] = []
                continue
            km = cluster_refinement(pts, self.threshold)
            lab = km.predict(pts)
            self.shapes[c] = [self.shape_cls(pts[lab == j])
                              for j in range(km.n_clusters) if (lab == j).any()]
        return self

    def accepts(self, V, preds):
        ok = np.zeros(len(V), dtype=bool)
        for c, shapes in self.shapes.items():
            idx = preds == c
            if idx.any() and shapes:
                ok[idx] = np.any([s.contains(V[idx], self.epsilon) for s in shapes], axis=0)
        return ok


class MultiLayerMonitor:
    """Original policy: accept only if EVERY watched layer accepts (max-confidence).

    shape_by_layer (NEW, optional): e.g. {-2: RotatedBox, -3: RotatedBox}.
    Layers not listed use the shape_cls given in **kw (default: Box).
    """
    def __init__(self, layers, shape_by_layer=None, **kw):
        self.layers = layers
        self.mons = {}
        for l in layers:
            kw_l = dict(kw)
            if shape_by_layer and l in shape_by_layer:
                kw_l["shape_cls"] = shape_by_layer[l]
            self.mons[l] = LayerMonitor(**kw_l)

    def fit(self, vals, y, n_classes):
        for l in self.layers:
            self.mons[l].fit(vals[l], y, n_classes)
        return self

    def accepts(self, vals, preds):
        return np.logical_and.reduce([self.mons[l].accepts(vals[l], preds) for l in self.layers])


class BooleanMonitor:
    """gamma = 0: accept iff the (activation > 0) pattern was seen for the predicted class."""
    def __init__(self):
        self.patterns = {}

    @staticmethod
    def _keys(H):
        return [r.tobytes() for r in np.packbits(H > 0, axis=1)]

    def fit(self, H, y, n_classes):
        for c in range(n_classes):
            self.patterns[c] = set(self._keys(H[y == c]))
        return self

    def accepts(self, H, preds):
        return np.array([k in self.patterns.get(int(p), ())
                         for k, p in zip(self._keys(H), preds)])