from abc import ABC, abstractmethod
import numpy as np
import scipy

class Kernel:

    def kernel_matrix(self, X):
        return self.kernel(*np.meshgrid(X, X))
    
    def features(self, X, Z):
        kernel_matrix = self.kernel_matrix(Z)
        K = scipy.linalg.sqrtm(np.linalg.pinv(kernel_matrix, hermitian=True))
        return K @ self.kernel(*np.meshgrid(Z, X))
    
    def mmd_nystrom_uniform(X, Y, a, p):
        W = np.concat(X, Y)
        n_x = np.shape(X)[0]
        n_y = np.shape(Y)[0]
        n = n_x + n_y

        rng = np.random.default_rng()
        Z = np.concat(rng.choice(X, n_x), rng.choice(Y, n_y))

        w_0 = np.concat(np.ones(n_x) / n_x, -np.ones(n_y) / n_y).reshape(1, n)
        w = np.repeat(w_0, p + 1, axis=0)
        rng.shuffle(w[1:])

        features = self.features(W, Z)
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





