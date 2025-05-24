import kernel
import numpy as np
import scipy
import matplotlib as plt
import sklearn

seed = 1048596
P = 200
alpha = 0.05
rng = np.random.default_rng(seed)
rbf = kernel.GaussianKernel()

gauss_dim = 1
sigma = 10
n_x, n_y = 2500, 2500
n = n_x + n_y
l_x, l_y = np.sqrt(n)//2, np.sqrt(n)//2

X = np.array(rng.normal(0, 1, n_x))
Y = np.array(rng.normal(0, sigma, n_y))

rbf.fit_bandwidth(X, Y, 1000)

rbf.test_mmd_nystrom_uniform(X, l_x, Y, l_y, alpha, P, seed)