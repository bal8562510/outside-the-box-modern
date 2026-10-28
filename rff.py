import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


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