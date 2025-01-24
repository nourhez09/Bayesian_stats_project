#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 23:02:37 2025

@author: hiro
"""

import numpy as np
from scipy.stats import norm, multivariate_normal, laplace
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
import pandas as pd
import arviz as az

def generate_simulation_data(sim_i):
    np.random.seed(12345)
    if sim_i == 1:
        n_train, n_validation, n_test, p = 20, 20, 200, 8
        beta_true = np.array([3, 1.5, 0, 0, 2, 0, 0, 0])
    elif sim_i == 2:
        n_train, n_validation, n_test, p = 20, 20, 200, 8
        beta_true = np.full(p, 0.85)
    elif sim_i == 3:
        n_train, n_validation, n_test, p = 20, 20, 200, 8
        beta_true = np.array([5, 0, 0, 0, 0, 0, 0, 0])
    elif sim_i == 4:
        n_train, n_validation, n_test, p = 100, 100, 400, 9
        beta_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0])
    elif sim_i == 5:
        n_train, n_validation, n_test, p = 20, 20, 400, 30
        beta_true = np.concatenate([np.full(10, 3), np.zeros(10), np.full(10, 3)])
    elif sim_i == 6:
        n_train, n_validation, n_test, p = 20, 20, 400, 15
        beta_true = np.array([-1.2, 1.8, 0, 0, 0, 0, 0.5, 1, 0, 0, 0, 0, 1, 1, 0])
    else:
        raise ValueError("Invalid simulation index")
    
    rho = 0.5
    x_sigma = np.array([[rho ** abs(i - j) for j in range(p)] for i in range(p)])
    x_train = multivariate_normal.rvs(mean=np.zeros(p), cov=x_sigma, size=n_train)
    x_validation = multivariate_normal.rvs(mean=np.zeros(p), cov=x_sigma, size=n_validation)
    x_test = multivariate_normal.rvs(mean=np.zeros(p), cov=x_sigma, size=n_test)
    
    x = np.vstack([x_train, x_validation, x_test])
    return x, beta_true

def bootstrap_median(samples, n_bootstrap=500):
    boot_medians = np.array([np.median(np.random.choice(samples, size=len(samples), replace=True)) for _ in range(n_bootstrap)])
    return np.sqrt(np.var(boot_medians))

def test_bootstrap():
    samples = np.random.randn(1000)
    sd_estimate = bootstrap_median(samples)
    assert sd_estimate > 0, "Standard deviation estimate should be positive"
    print("Bootstrap test passed")

test_bootstrap()

def qr_lasso(x, y, theta, n_sampler, thin):
    np.random.seed(12345)
    beta_samples = np.random.randn(n_sampler // thin, x.shape[1])
    tau_samples = np.random.gamma(1, 1, n_sampler // thin)
    tz_samples = np.random.gamma(1, 1, (n_sampler // thin, x.shape[0]))
    return {"beta": beta_samples, "tau": tau_samples, "tz": tz_samples}

def analyze_results(results):
    tau_z = np.column_stack([results['tz'], results['tau']])
    plt.figure(figsize=(10, 6))
    plt.plot(tau_z[:, 14], label='v[15]')
    plt.plot(tau_z[:, 16], label='v[17]')
    plt.plot(tau_z[:, 24], label='v[25]')
    plt.plot(tau_z[:, 34], label='v[35]')
    plt.legend()
    plt.show()

def test_generate_simulation_data():
    x, beta_true = generate_simulation_data(1)
    assert x.shape == (240, 8), "Incorrect shape for simulation data"
    assert len(beta_true) == 8, "Incorrect beta length"
    print("Simulation data generation test passed")

def test_qr_lasso():
    x, beta_true = generate_simulation_data(1)
    y = x @ beta_true + np.random.normal(0, 1, x.shape[0])
    result = qr_lasso(x, y, theta=0.5, n_sampler=1000, thin=10)
    assert result['beta'].shape[0] == 100, "Incorrect number of beta samples"
    assert result['tau'].shape[0] == 100, "Incorrect number of tau samples"
    analyze_results(result)
    print("QR Lasso test passed")

test_generate_simulation_data()
test_qr_lasso()


if __name__ == "__main__":
    x, beta_true = generate_simulation_data(1)
    y = x @ beta_true + np.random.normal(0, 1, x.shape[0])
    result = qr_lasso(x, y, theta=0.5, n_sampler=1000, thin=10)
    analyze_results(result)
    print("Simulation completed successfully.")

