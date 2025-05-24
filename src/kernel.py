from abc import ABC, abstractmethod
import numpy as np
import scipy
from sklearn.gaussian_process.kernels import RBF
import johnson


def nondiag(X):
    Y = X.copy()
    np.fill_diagonal(Y, 0)
    return Y


class Kernel(ABC):

    @abstractmethod
    def __call__(self, X, Y):
        raise NotImplementedError
    
    def features(self, X, Z):
        kernel_matrix = self(Z, Z)
        K = scipy.linalg.sqrtm(np.linalg.pinv(kernel_matrix, hermitian=True))
        return K @ self(Z, X)
    
    def h_kernel(self, X, Y):
        cross_matrix = self(X, Y)
        return self(X) + self(Y) - cross_matrix - cross_matrix.T
    
    def mmd_U(self, X, Y):
        """
        Computes an unbiased estimator of the squared MMD of two samples.
        Based on Gretton 2012, p.6 - 7 (lemma 6).
        Assumes X and Y have the same shape.
        """
        m = np.shape(X)[0]
        h_matrix = self.h_kernel(X, Y)
        return m * nondiag(h_matrix).sum() / ((m * (m - 1)))
    
    def moments_mmd_U(self, X, Y):
        """
        Computes the moment of the MMD U.
        Gretton 2012 p.15 (11), (12), (13)
        """
        m = np.shape(X)[0]
        h_matrix = self.h_kernel(X, Y)
        sq_h_matrix = h_matrix ** 2

        variance = 2 / (m*(m-1)) * nondiag(sq_h_matrix).sum() / ((m-1) * m)
        variance *= m**2

        int_exp = h_matrix @ h_matrix.T - np.diag(h_matrix) * h_matrix - np.diag(np.array(h_matrix)).T * h_matrix
        int_exp = int_exp / (m - 2)
        moment_3 = 8 * (m - 2) / (m**2 * (m - 1) ** 2) * nondiag(self.h_kernel(X, Y) * int_exp).sum() / (m * (m - 1))
        moment_3 *= m**3

        skewness = moment_3 / variance ** (3 / 2)
        kurtosis = 2 * (skewness ** 2 + 1)

        return [0, variance, skewness, kurtosis]    


    def test_mmd_U(self, X, Y, a, pval=False):
        johnson_dist = johnson.fit_johnsonsu_by_moments(*self.moments_mmd_U(X, Y))
        test_stat = self.mmd_U(X, Y)
        if pval:
            return johnson_dist.cdf(test_stat)
        return self.mmd_U(X, Y) <= johnson_dist.ppf(1-a)
    
    def test_mmd_nystrom_uniform(self, X, Y, a, P, pval=False):
        W = np.concat(X, Y)
        n_x = np.shape(X)[0]
        n_y = np.shape(Y)[0]
        n = n_x + n_y
        l = n

        rng = np.random.default_rng()
        Z = np.concat(rng.choice(X, n_x), rng.choice(Y, n_y))

        w_0 = np.concat(np.ones(n_x) / n_x, -np.ones(n_y) / n_y).reshape(1, n)
        w = np.repeat(w_0, P + 1, axis=0)
        rng.shuffle(w[1:])

        features = self.features(W, Z).T
        v = np.zeros((P, l))
        for i in range(n):
            for p in range(P):
                v[p] = w[p, i] * features[i]
        psi = np.linalg.norm(v, axis=1)
        b_a = np.floor((1-a)*(p+1))
        psi_b_a = np.partition(psi, b_a)[b_a]


        if psi[0] > psi_b_a:
            return 1
        elif psi[0] == psi_b_a:
            psi_eq = np.count_nonzero(psi == psi_b_a)
            psi_gt = np.count_nonzero(psi > psi_b_a)
            p = (a*(p+1)-psi_gt) / psi_eq
            return rng.binomial(1, p)
        return 0


class GaussianKernel(RBF, Kernel):

    def __init__(self, *args, **kwargs):
        RBF.__init__(self, *args, **kwargs)
    
    def auto_length(X, n):
        pass