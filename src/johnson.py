from scipy.stats import johnsonsu
from scipy.optimize import minimize
import numpy as np
import scipy.stats as stats


def fit_johnsonsu_by_moments(mean, var, skew, kurt):
    def objective(params):
        gamma, delta, loc, scale = params
        m = johnsonsu.mean(gamma, delta, loc, scale)
        v = johnsonsu.var(gamma, delta, loc, scale)
        s = johnsonsu.stats(gamma, delta, moments='s')
        k = johnsonsu.stats(gamma, delta, moments='k')
        return (m - mean)**2 + (v - var)**2 + (s - skew)**2 + (k - kurt)**2

    initial_guess = [1.0, 1.0, mean, np.sqrt(var)]
    result = minimize(objective, initial_guess, method='L-BFGS-B')
    return johnsonsu(*result.x)
