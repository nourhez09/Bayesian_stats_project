#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 23:02:37 2025

@author: hiro
"""
import numpy as np
from scipy.stats import invgauss, gamma, norm

def qr_lasso(x, y, theta=0.5, n_sampler=13000, n_burn=3000, thin=20):
    n, p = x.shape
    xi1 = (1 - 2 * theta) / (theta * (1 - theta))
    xi2 = np.sqrt(2 / (theta * (1 - theta)))

    # Priors
    a, b, c, d = 1e-1, 1e-1, 1e-1, 1e-1

    # Initialization
    beta_c = np.ones(p)
    tz_c = np.ones(n)
    s_c = np.ones(p)
    tau_c = 1
    eta2_c = 1

    # Storing sampled values
    beta_p = np.zeros((n_sampler, p))
    tz_p = np.zeros((n_sampler, n))
    tau_p = np.zeros(n_sampler)
    eta2_p = np.zeros(n_sampler)

    for iter in range(n_sampler):
        if iter % 1000 == 0:
            print(f"This is step {iter}")

        # Update tz
        temp_lambda = xi1 ** 2 * tau_c / (xi2 ** 2) + 2 * tau_c
        temp_nu = np.sqrt(temp_lambda * xi2 ** 2 / (tau_c * (y - x @ beta_c) ** 2))
        temp_tz = invgauss.rvs(temp_lambda / temp_nu, scale=temp_nu)
        tz_c[temp_tz > 0] = 1 / temp_tz[temp_tz > 0]

        # Update s
        temp_lambda = eta2_c
        temp_nu = np.sqrt(temp_lambda / beta_c ** 2)
        temp_s = invgauss.rvs(temp_lambda / temp_nu, scale=temp_nu)
        s_c[temp_s > 0] = 1 / temp_s[temp_s > 0]

        # Update beta
        for k in range(p):
            temp_var = 1 / (np.sum(x[:, k] ** 2 * tau_c / (xi2 ** 2 * tz_c)) + 1 / s_c[k])
            temp_mean = temp_var * np.sum(x[:, k] * (y - xi1 * tz_c - x @ beta_c + x[:, k] * beta_c[k]) * tau_c / (xi2 ** 2 * tz_c))
            beta_c[k] = norm.rvs(temp_mean, np.sqrt(temp_var))

        # Update tau
        temp_shape = a + 3 / 2 * n
        temp_rate = np.sum((y - x @ beta_c - xi1 * tz_c) ** 2 / (2 * xi2 ** 2 * tz_c) + tz_c) + b
        tau_c = gamma.rvs(temp_shape, scale=1 / temp_rate)

        # Update eta2
        temp_shape = p + c
        temp_rate = np.sum(s_c) / 2 + d
        eta2_c = gamma.rvs(temp_shape, scale=1 / temp_rate)

        beta_p[iter, :] = beta_c
        tz_p[iter, :] = tz_c
        tau_p[iter] = tau_c
        eta2_p[iter] = eta2_c

    temp_indices = np.arange(n_burn, n_sampler, thin)
    result = {
        'beta': beta_p[temp_indices, :],
        'tz': tz_p[temp_indices, :],
        'tau': tau_p[temp_indices],
        'eta2': eta2_p[temp_indices]
    }
    return result

# Example test to validate the function
if __name__ == "__main__":
    np.random.seed(42)
    x_test = np.random.normal(0, 1, (50, 5))
    beta_true = np.array([1.5, -2.0, 3.0, 0.0, 0.5])
    y_test = x_test @ beta_true + np.random.normal(0, 1, 50)
    result = qr_lasso(x_test, y_test, theta=0.5, n_sampler=500, n_burn=100, thin=10)
    print("Estimated beta:", np.mean(result['beta'], axis=0))

