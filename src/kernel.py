from abc import ABC, abstractmethod
import numpy as np
import scipy
from sklearn.gaussian_process.kernels import RBF
import johnson
from scipy.spatial.distance import pdist, squareform

def nondiagsum(X):
    return np.sum(X) - np.trace(X)


def decision(psi, alpha, rng, pval=False):
    P = psi.shape[0]
    b_a = int((1 - alpha) * (P + 1))
    psi_b_a = np.partition(psi, b_a)[b_a]

    if pval:
        psi_eq = np.count_nonzero(psi == psi[0])
        psi_gt = np.count_nonzero(psi > psi[0])
        u = rng.uniform()
        return (psi_gt + psi_eq * u) / (P + 1)

    if psi[0] > psi_b_a:
        return 1

    elif psi[0] == psi_b_a:
        psi_eq = np.count_nonzero(psi == psi_b_a)
        psi_gt = np.count_nonzero(psi > psi_b_a)
        p = (alpha * (P + 1) - psi_gt) / psi_eq
        u = rng.uniform(0, 1)
        return int(u < p)

    return 0


class Kernel(ABC):

    @abstractmethod
    def __call__(self, X, Y):
        raise NotImplementedError

    def nystrom_features(self, X, inds):
        kernel_matrix = self(X, X[inds])
        K = kernel_matrix[inds, :]  # = K(X[inds], X[inds])
        eigen_vals, eigen_vecs = np.linalg.eigh(K)
        # Ensure eigenvalues are positive (numerical stability)
        tol = np.finfo(eigen_vals.dtype).eps * 10  # Tolerance for positivity
        adjusted_vals = np.maximum(eigen_vals, tol)
        sqrt_inv_vals = 1.0 / np.sqrt(adjusted_vals)
        sqrt_inv_K = (eigen_vecs * sqrt_inv_vals) @ eigen_vecs.T

        return sqrt_inv_K @ kernel_matrix.T

    def test_mmd_nystrom_uniform(self, X, l_x, Y, l_y, alpha, P, rng, pval=False):
        W = np.concat((X, Y))
        n_x = np.shape(X)[0]
        n_y = np.shape(Y)[0]
        n = n_x + n_y
        l = l_x + l_y

        inds = np.concat((rng.choice(n_x, size=l_x, replace=False), rng.choice(n_y, size=l_y, replace=False) + n_x))

        w_0 = np.concat((np.ones(n_x) / n_x, -np.ones(n_y) / n_y)).reshape(1, n)
        w = np.repeat(w_0, P + 1, axis=0)
        rng.shuffle(w[1:], axis=1)

        features = self.nystrom_features(W, inds)
        v = w @ features.T
        psi = np.sum(v**2, axis=1)

        return decision(psi, alpha, rng, pval)

    def h_kernel(self, X, Y):
        K_XY = self(X, Y)
        return self(X) + self(Y) - K_XY - K_XY.T

    def MMD_2_U(self, X, Y):
        """
        Computes an unbiased estimator of the squared MMD of two samples.
        Based on Gretton 2012, p.6 - 7 (lemma 6).
        Assumes X and Y have the same shape.
        """
        m = np.shape(X)[0]
        h_matrix = self.h_kernel(X, Y)
        return m * nondiagsum(h_matrix) / ((m * (m - 1)))

    def moments_MMD_2_U(self, X, Y):
        """
        Computes the moment of the MMD U.
        Gretton 2012 p.15 (11), (12), (13)
        """
        m = np.shape(X)[0]
        K_XY = self.h_kernel(X, Y)

        variance = 2 / (m * (m - 1)) * nondiagsum(K_XY**2) / ((m - 1) * m)
        variance *= m**2

        int_exp = (
            K_XY @ K_XY.T
            - np.diag(K_XY) * K_XY
            - np.diag(np.array(K_XY)).T * K_XY
        )
        int_exp = int_exp / (m - 2)
        moment_3 = (
            8
            * (m - 2)
            / (m**2 * (m - 1) ** 2)
            * nondiagsum(K_XY * int_exp)
            / (m * (m - 1))
        )
        moment_3 *= m**3

        skewness = moment_3 / variance ** (3 / 2)
        kurtosis = 2 * (skewness**2 + 1)

        return [0, variance, skewness, kurtosis]

    def test_MMD_2_U_M(self, X, Y, alpha, pval=False):
        johnson_dist = johnson.fit_johnsonsu_by_moments(*list(self.moments_MMD_2_U(X, Y)))
        test_stat = self.MMD_2_U(X, Y)
        if pval:
            return johnson_dist.cdf(test_stat)
        return int(self.MMD_2_U(X, Y) > johnson_dist.ppf(1 - alpha))

    def test_MMD_2_U_B(self, X, Y, alpha, P, rng=None, pval=False):
        if not rng:
            rng = np.random.default_rng()
        Z = np.concat((X, Y))
        MMDs = np.zeros(P + 1)
        for k in range(P + 1):
            MMDs[k] = self.MMD_2_U(Z[: X.shape[0]], Z[X.shape[0] :])
            rng.shuffle(Z)
        return decision(MMDs, alpha, rng, pval)


class GaussianKernel(RBF, Kernel):

    def __init__(self, *args, **kwargs):
        RBF.__init__(self, *args, **kwargs)

    def fit_bandwidth(self, X, Y, n):
        Z = np.concat((X[: n // 2], Y[: n // 2]))
        self.length_scale = np.median(scipy.spatial.distance.pdist(Z))