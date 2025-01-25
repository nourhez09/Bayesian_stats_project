
################################################################################################################
# Modification 1: adding all regularized versions to the 1. 
# The group-lasso approach is only approximate. A full Bayesian group-lasso might require specialized “multivariate normal” draws.
# 1. [functions.R]

import numpy as np
from numpy.random import default_rng
from scipy.stats import invgauss, gamma, norm, multivariate_normal
from sklearn.model_selection import KFold
from sklearn.linear_model import LassoLarsCV, ElasticNetCV, LassoCV

##############################################################################
# 1. QR with Lasso prior (qr.lasso)
##############################################################################
def qr_lasso(x, y, theta=0.5, n_sampler=13000, n_burn=3000, thin=20, seed=None):
    """
    Approximate Python version of the R function qr.lasso.
    Model:
      y[i] = x[i]^T * beta + xi1 * tz[i] + tau^(-0.5)*xi2*sqrt(tz[i])*N(0,1)
      tz[i] ~ InverseGauss(...)
    """
    rng = default_rng(seed)

    n, p = x.shape
    xi1 = (1 - 2*theta) / (theta*(1 - theta))
    xi2 = np.sqrt(2 / (theta * (1 - theta)))

    # Priors
    a = 1e-1
    b_ = 1e-1
    c_ = 1e-1
    d_ = 1e-1

    # Initialization
    beta_c = np.ones(p)
    tz_c = np.ones(n)
    s_c = np.ones(p)
    tau_c = 1.0
    eta2_c = 1.0

    # Storage arrays
    beta_p = np.zeros((n_sampler, p))
    tz_p = np.zeros((n_sampler, n))
    tau_p = np.zeros(n_sampler)
    eta2_p = np.zeros(n_sampler)

    for i in range(n_sampler):
        if i % 1000 == 0:
            print(f"qr_lasso: This is step {i}")

        # --- tz update (Inverse Gaussian draws)
        temp_lambda = xi1**2 * tau_c / (xi2**2) + 2 * tau_c
        temp_nu = np.sqrt(temp_lambda * xi2**2 / (tau_c * (y - x @ beta_c)**2))

        tz_draw = np.full(n, np.nan)
        indices = np.arange(n)
        while len(indices) > 0:
            candidate = invgauss.rvs(mu=temp_lambda / temp_nu[indices],
                                     scale=temp_nu[indices],
                                     size=len(indices),
                                     random_state=rng)
            valid_mask = (candidate > 0) & (~np.isnan(candidate))
            tz_draw[indices[valid_mask]] = 1.0 / candidate[valid_mask]
            indices = indices[~valid_mask]
        tz_c = tz_draw

        # --- s update
        # s.c ~ InverseGauss(eta2, sqrt(eta2/beta^2)?)
        temp_lambda = eta2_c
        temp_nu = np.sqrt(temp_lambda / (beta_c**2))

        s_draw = np.full(p, np.nan)
        indices = np.arange(p)
        while len(indices) > 0:
            candidate = invgauss.rvs(mu=temp_lambda / temp_nu[indices],
                                     scale=temp_nu[indices],
                                     size=len(indices),
                                     random_state=rng)
            valid_mask = (candidate > 0) & (~np.isnan(candidate))
            s_draw[indices[valid_mask]] = 1.0 / candidate[valid_mask]
            indices = indices[~valid_mask]
        s_c = s_draw

        # --- beta update
        for k in range(p):
            temp_var_inv = np.sum(x[:, k]**2 * tau_c / (xi2**2 * tz_c)) + 1.0/s_c[k]
            temp_var = 1.0 / temp_var_inv
            # remove x[:,k]*beta_c[k] from residual
            partial_resid = y - x @ beta_c + x[:, k]*beta_c[k]
            temp_mean = temp_var * np.sum(x[:, k] * (partial_resid - xi1*tz_c)
                                          * (tau_c / (xi2**2 * tz_c)))
            beta_c[k] = rng.normal(loc=temp_mean, scale=np.sqrt(temp_var))

        # --- tau update
        temp_shape = a + 1.5 * n
        temp_rate = np.sum(((y - x @ beta_c - xi1*tz_c)**2)/(2*xi2**2*tz_c) + tz_c) + b_
        tau_c = rng.gamma(shape=temp_shape, scale=1.0 / temp_rate)

        # --- eta2 update
        temp_shape = p + c_
        temp_rate = np.sum(s_c)/2.0 + d_
        eta2_c = rng.gamma(shape=temp_shape, scale=1.0/temp_rate)

        # Store
        beta_p[i] = beta_c
        tz_p[i] = tz_c
        tau_p[i] = tau_c
        eta2_p[i] = eta2_c

    # Post-processing
    idx = np.arange(n_burn, n_sampler, thin)
    result = {
        'beta': beta_p[idx],
        'tz': tz_p[idx],
        'tau': tau_p[idx],
        'eta2': eta2_p[idx]
    }
    return result


##############################################################################
# 2. QR with Group Lasso (qr.grouplasso)
##############################################################################
def qr_grouplasso(x, y, theta=0.5, group=None, K=None,
                  n_sampler=15000, n_burn=5000, thin=20, seed=None):
    """
    Approximates the R 'qr.grouplasso' approach using group-based sampling.
    group: list of arrays specifying which columns of x are in each group.
    K:     optional list of matrices for each group.
    """
    rng = default_rng(seed)
    n, p = x.shape
    xi1 = (1 - 2*theta)/(theta*(1-theta))
    xi2 = np.sqrt(2/(theta*(1-theta)))

    if group is None:
        group = [np.arange(p)]
    G = len(group)

    if K is None:
        K = []
        for grp in group:
            grp_len = len(grp)
            K.append(np.eye(grp_len)*grp_len)

    # Priors
    a, b_, c_, d_ = 1e-1, 1e-1, 1e-1, 1e-1

    # Init
    beta_c = np.ones(p)
    tz_c = np.ones(n)
    s_c = np.ones(G)
    tau_c = 1.0
    eta2_c = 1.0

    beta_list = []
    tau_list = []
    eta2_list = []

    for i in range(n_sampler):
        if i % 1000 == 0:
            print(f"qr_grouplasso: step {i}")

        # tz
        temp_lambda = xi1**2 * tau_c / xi2**2 + 2.0*tau_c
        temp_nu = np.sqrt(temp_lambda * xi2**2/(tau_c*(y - x @ beta_c)**2))

        tz_draw = np.full(n, np.nan)
        idx_ = np.arange(n)
        while len(idx_)>0:
            candidate = invgauss.rvs(mu=temp_lambda/temp_nu[idx_],
                                     scale=temp_nu[idx_],
                                     size=len(idx_),
                                     random_state=rng)
            valid = (candidate>0)&(~np.isnan(candidate))
            tz_draw[idx_[valid]] = 1.0 / candidate[valid]
            idx_ = idx_[~valid]
        tz_c = tz_draw

        # s
        temp_lambda = eta2_c
        for g_idx in range(G):
            grp_cols = group[g_idx]
            betag = beta_c[grp_cols]
            # betag^T * K[g] * betag
            val = betag @ (K[g_idx] @ betag)
            tmp_nu = np.sqrt(temp_lambda/val)
            success = False
            while not success:
                candidate = invgauss.rvs(mu=temp_lambda/tmp_nu,
                                         scale=tmp_nu,
                                         size=1,
                                         random_state=rng)[0]
                if candidate>0 and not np.isnan(candidate):
                    s_c[g_idx] = 1.0/candidate
                    success = True

        # beta
        tz_inv = 1.0/tz_c
        for g_idx in range(G):
            grp_cols = group[g_idx]
            Xg = x[:, grp_cols]
            temp_mat = (tz_inv[:, None]*Xg).T @ Xg
            temp_var = np.linalg.inv(tau_c*xi2**(-2)*temp_mat + (K[g_idx]/s_c[g_idx]))

            # partial residual
            other_cols = np.setdiff1d(np.arange(p), grp_cols)
            yg = y - xi1*tz_c - x[:, other_cols]@beta_c[other_cols]
            tmp_vec = (tz_inv*yg) @ Xg
            temp_mean = (tau_c/(xi2**2))*(temp_var@tmp_vec)
            # sample from MVN
            beta_c[grp_cols] = multivariate_normal.rvs(mean=temp_mean,
                                                       cov=temp_var,
                                                       random_state=rng)

        # tau
        shape_ = a + 1.5*n
        rate_ = np.sum(((y - x@beta_c - xi1*tz_c)**2)/(2*xi2**2*tz_c) + tz_c)+b_
        tau_c = rng.gamma(shape=shape_, scale=1.0/rate_)

        # eta2
        shape_ = (p+G)/2.0 + c_
        rate_ = np.sum(s_c)/2.0 + d_
        eta2_c = rng.gamma(shape=shape_, scale=1.0/rate_)

        beta_list.append(beta_c.copy())
        tau_list.append(tau_c)
        eta2_list.append(eta2_c)

    idx_ = np.arange(n_burn, n_sampler, thin)
    beta_list = np.array(beta_list)[idx_]
    tau_list = np.array(tau_list)[idx_]
    eta2_list = np.array(eta2_list)[idx_]

    return {
        'beta': beta_list,
        'tau': tau_list,
        'eta2': eta2_list
    }


##############################################################################
# 3. QR with Elastic Net prior (qr.enet)
##############################################################################
def qr_enet(x, y, theta=0.5, n_sampler=13000, n_burn=3000, thin=20, seed=None):
    """
    Approximates R's qr.enet function
    """
    rng = default_rng(seed)
    n, p = x.shape
    xi1 = (1 - 2*theta)/(theta*(1-theta))
    xi2 = np.sqrt(2/(theta*(1-theta)))

    # Priors
    a, b_ = 1e-1, 1e-1
    c1, d1 = 1e-1, 1e-1
    c2, d2 = 1e-1, 1e-1

    beta_c = np.ones(p)
    tz_c = np.ones(n)
    t_c = np.ones(p)
    tau_c = 1.0
    eta2_c = 1.0
    teta1_c = 1.0

    tz_list = []
    beta_list = []
    tau_list = []
    eta2_list = []
    teta1_list = []

    rejections = 0

    for i in range(n_sampler):
        if i%1000==0:
            print(f"qr_enet: step {i}")

        # tz
        temp_lambda = xi1**2 * tau_c/(xi2**2)+2*tau_c
        temp_nu = np.sqrt(temp_lambda*xi2**2/(tau_c*(y - x@beta_c)**2))
        tz_draw = np.full(n, np.nan)
        idx_ = np.arange(n)
        while len(idx_)>0:
            candidate = invgauss.rvs(mu=temp_lambda/temp_nu[idx_],
                                     scale=temp_nu[idx_],
                                     size=len(idx_),
                                     random_state=rng)
            valid = (candidate>0)&(~np.isnan(candidate))
            tz_draw[idx_[valid]] = 1.0/candidate[valid]
            idx_ = idx_[~valid]
        tz_c = tz_draw

        # t
        # t.c ~ InverseGauss( 2*teta1, sqrt(2*teta1/(2*eta2*beta^2)) ) => 1/candidate+1
        temp_lambda_2 = 2.0*teta1_c
        temp_nu_2 = np.sqrt(temp_lambda_2/(2*eta2_c*(beta_c**2)))

        t_draw = np.full(p, np.nan)
        idx_ = np.arange(p)
        while len(idx_)>0:
            candidate = invgauss.rvs(mu=temp_lambda_2/temp_nu_2[idx_],
                                     scale=temp_nu_2[idx_],
                                     size=len(idx_),
                                     random_state=rng)
            valid = (candidate>0)&(~np.isnan(candidate))
            t_draw[idx_[valid]] = 1.0/candidate[valid]+1.0
            idx_ = idx_[~valid]
        t_c = t_draw

        # beta
        for k in range(p):
            # var^-1 = tau_c/(xi2^2)* sum(x[:,k]^2/tz_c) + 2*eta2_c * t[k]/(t[k]-1)
            temp_var_inv = tau_c/(xi2**2)*np.sum((x[:, k]**2)/tz_c) + 2*eta2_c*(t_c[k]/(t_c[k]-1))
            temp_var = 1.0/temp_var_inv

            # partial residual
            partial_res = y - x@beta_c + x[:,k]*beta_c[k]
            temp_mean = temp_var * (tau_c/(xi2**2))*np.sum(x[:,k]*(partial_res - xi1*tz_c)/tz_c)
            beta_c[k] = rng.normal(loc=temp_mean, scale=np.sqrt(temp_var))

        # tau
        shape_ = a+1.5*n
        rate_ = np.sum(((y - x@beta_c - xi1*tz_c)**2)/(2*xi2**2*tz_c) + tz_c)+b_
        tau_c = rng.gamma(shape=shape_, scale=1.0/rate_)

        # teta1 (metropolis step in R, we do a placeholder)
        shape_ = p + c1
        rate_ = np.sum(t_c-1)+d1
        temp_teta1 = rng.gamma(shape=shape_, scale=1.0/rate_)
        # the log acceptance ratio in R is complicated; we put a placeholder
        acceptance_ratio = 0.0
        log_u = np.log(rng.uniform())
        if log_u <= min(acceptance_ratio, 0.0):
            teta1_c = temp_teta1
        else:
            rejections+=1

        # eta2
        shape_ = p/2.0 + c2
        rate_ = np.sum((t_c/(t_c-1))*(beta_c**2)) + d2
        eta2_c = rng.gamma(shape=shape_, scale=1.0/rate_)

        tz_list.append(tz_c.copy())
        beta_list.append(beta_c.copy())
        tau_list.append(tau_c)
        eta2_list.append(eta2_c)
        teta1_list.append(teta1_c)

    idx_ = np.arange(n_burn, n_sampler, thin)
    tz_list = np.array(tz_list)[idx_]
    beta_list = np.array(beta_list)[idx_]
    tau_list = np.array(tau_list)[idx_]
    eta2_list = np.array(eta2_list)[idx_]
    teta1_list = np.array(teta1_list)[idx_]

    return {
        'beta': beta_list,
        'tz': tz_list,
        'tau': tau_list,
        'eta2': eta2_list,
        'teta1': teta1_list,
        'rejections': rejections
    }


##############################################################################
# 4. check(x, theta)
##############################################################################
def check_residuals(x, theta):
    """
    R code:
      result = (x > 0)*x*theta + (x <= 0)*(-1+theta)*x
    """
    return np.where(x>0, x*theta, x*(theta-1.0))


##############################################################################
# 5. LARSL
##############################################################################
def LARSL(x, y, K=10):
    """
    Approximate R's LARSL using LassoLarsCV from scikit-learn
    """
    model = LassoLarsCV(cv=K).fit(x, y)
    beta = model.coef_
    s = model.alpha_
    # approximate error: MSE path
    err = np.min(model.mse_path_.mean(axis=1))
    return {
        's': s,
        'beta': beta,
        'err': err
    }

##############################################################################
# 6. LARSLV
##############################################################################
def LARSLV(x_train, y_train, x_val, y_val):
    """
    Approximate R's LARSLV:
      - Fit LARS on (x_train,y_train).
      - Evaluate a grid to pick best fraction => we emulate with LassoLarsCV
    """
    # scikit-learn doesn't let us easily step fraction from 0..1
    # We'll do a direct approach
    model = LassoLarsCV(cv=5).fit(x_train, y_train)
    # Evaluate on validation
    y_pred = model.predict(x_val)
    mse = np.mean((y_val - y_pred)**2)
    # Refit on combined
    x_comb = np.vstack((x_train, x_val))
    y_comb = np.concatenate((y_train, y_val))
    final_model = LassoLarsCV(cv=5).fit(x_comb, y_comb)
    return {
        's': final_model.alpha_,
        'beta': final_model.coef_,
        'err': mse
    }

##############################################################################
# 7. ENL
##############################################################################
def ENL(x, y, lambda2_list=(0,0.01,0.1,1,10,100,1000), K=10):
    """
    R code picks best L2 penalty among a pool, does K-fold CV for L1 part.
    We'll do a rough approach using ElasticNetCV, looping over possible "lambda2".
    We'll treat 'lambda2' as controlling (1-l1_ratio).
    """
    best_err = float('inf')
    best_lambda2 = None
    best_beta = None
    n, p = x.shape

    for lam2 in lambda2_list:
        # interpret lam2 as ratio => l1_ratio=1/(1+lam2)
        l1_ratio = 1.0/(1.0+lam2)
        model = ElasticNetCV(l1_ratio=[l1_ratio], cv=K).fit(x, y)
        fold_mse = np.min(model.mse_path_.mean(axis=1))
        if fold_mse<best_err:
            best_err = fold_mse
            best_lambda2 = lam2
            best_beta = model.coef_.copy()

    return {
        's': None,  # no direct fraction akin to R
        'lambda2': best_lambda2,
        'beta': best_beta,
        'err': best_err
    }

##############################################################################
# 8. ENLV
##############################################################################
def ENLV(x_train, y_train, x_val, y_val,
         lambda2_list=(0,0.01,0.1,1,10,100,1000)):
    """
    Similar to ENL but pick best L2 penalty by a separate validation set
    """
    best_err = float('inf')
    best_lambda2 = None
    best_beta = None

    for lam2 in lambda2_list:
        l1_ratio = 1.0/(1.0+lam2)
        model = ElasticNetCV(l1_ratio=[l1_ratio], cv=5).fit(x_train, y_train)
        y_pred = model.predict(x_val)
        val_mse = np.mean((y_val - y_pred)**2)
        if val_mse<best_err:
            best_err = val_mse
            best_lambda2 = lam2
            best_beta = model.coef_.copy()

    # Refit on combined data
    x_comb = np.vstack((x_train, x_val))
    y_comb = np.concatenate((y_train, y_val))
    final_l1_ratio = 1.0/(1.0+best_lambda2)
    final_model = ElasticNetCV(l1_ratio=[final_l1_ratio], cv=5).fit(x_comb, y_comb)
    final_beta = final_model.coef_.copy()
    return {
        's': None,
        'lambda2': best_lambda2,
        'beta': final_beta,
        'err': best_err
    }

##############################################################################
# 9. qrL1CV and qrL1V
##############################################################################
# The original R code references a custom "QReg" function. We'll do placeholders.

def qrL1CV(x, y, theta=0.5, K=10, seed=None):
    """
    Emulates the R logic: uses K-fold CV to pick an 's' 
    for the custom QReg approach, then refits. 
    We'll approximate QReg with placeholders.
    """
    rng = default_rng(seed)
    n = len(y)
    indices = np.arange(n)
    rng.shuffle(indices)
    folds = np.array_split(indices, K)

    def QReg(x_train, y_train, tau=0.5):
        # returns a path => "S", "Beta0", "Beta"
        return {
            "S": np.linspace(1, 100, 10),
            "Beta0": np.linspace(0, 1, 10),
            "Beta": np.random.normal(size=(10, x_train.shape[1]))
        }

    def predQReg(x_test, y_test, fit_res):
        # compute resL2 for each step
        Beta_all = fit_res["Beta"]
        Beta0_all = fit_res["Beta0"]
        resL2 = []
        for i in range(len(Beta_all)):
            pred = Beta0_all[i] + x_test @ Beta_all[i]
            mse = np.mean((y_test - pred)**2)
            resL2.append(mse)
        return {"resL2": np.array(resL2)}

    s_min = []
    for i in range(K):
        omit = folds[i]
        train_idx = np.setdiff1d(np.arange(n), omit)
        x_train_sub, y_train_sub = x[train_idx], y[train_idx]
        x_test_sub, y_test_sub = x[omit], y[omit]

        # Add tiny noise as in R
        y_train_noisy = y_train_sub + rng.normal(scale=1e-8, size=len(y_train_sub))

        fit_res = QReg(x_train_sub, y_train_noisy, tau=theta)
        pred_res = predQReg(x_test_sub, y_test_sub, fit_res)
        idx_min = np.argmin(pred_res["resL2"])
        s_min.append(fit_res["S"][idx_min])

    s_choose = np.mean(s_min)
    # Re-fit on all data
    y_noisy = y + rng.normal(scale=1e-8, size=len(y))
    fit_full = QReg(x, y_noisy, tau=theta)
    # pick s closest to s_choose
    S_array = fit_full["S"]
    idx_min = np.argmin(np.abs(S_array - s_choose))
    beta0 = fit_full["Beta0"][idx_min]
    beta = fit_full["Beta"][idx_min]
    return {
        "beta0": beta0,
        "beta": beta
    }


def qrL1V(x_train, y_train, x_val, y_val, theta=0.5, seed=None):
    """
    Similar logic but uses a single validation set
    """
    rng = default_rng(seed)
    def QReg(x_train_, y_train_, tau=0.5):
        return {
            "S": np.linspace(1, 100, 10),
            "Beta0": np.linspace(0, 1, 10),
            "Beta": np.random.normal(size=(10, x_train_.shape[1]))
        }
    def predQReg(x_test_, y_test_, fit_res):
        Beta_all = fit_res["Beta"]
        Beta0_all = fit_res["Beta0"]
        resL2 = []
        for i in range(len(Beta_all)):
            pred = Beta0_all[i] + x_test_ @ Beta_all[i]
            mse = np.mean((y_test_ - pred)**2)
            resL2.append(mse)
        return {"resL2": np.array(resL2)}

    fit_train = QReg(x_train, y_train, tau=theta)
    pred_res = predQReg(x_val, y_val, fit_train)
    idx_min = np.argmin(pred_res["resL2"])
    s_choose = fit_train["S"][idx_min]

    # combine
    x_comb = np.vstack((x_train, x_val))
    y_comb = np.concatenate((y_train, y_val))
    fit_comb = QReg(x_comb, y_comb, tau=theta)
    S_array = fit_comb["S"]
    idx_ = np.argmin(np.abs(S_array - s_choose))
    beta0 = fit_comb["Beta0"][idx_]
    beta = fit_comb["Beta"][idx_]
    return {
        "beta0": beta0,
        "beta": beta
    }


##############################################################################
# EXAMPLE USAGE
##############################################################################
if __name__ == "__main__":
    # Example usage for qr_lasso
    np.random.seed(42)
    X_example = np.random.normal(size=(50, 5))
    beta_true = np.array([1.5, -2.0, 3.0, 0.0, 0.5])
    y_example = X_example @ beta_true + np.random.normal(size=50)

    print("Running qr_lasso on small example:")
    result_lasso = qr_lasso(X_example, y_example, theta=0.5, n_sampler=1000, n_burn=200, thin=10)
    print("Sampled betas shape:", result_lasso["beta"].shape)

    # Example usage for LARSL
    print("\nRunning LARSL (sklearn LassoLarsCV) on small example:")
    lars_out = LARSL(X_example, y_example)
    print(f"LARSL best alpha (s): {lars_out['s']}, Coeffs: {lars_out['beta']}, CV error approx: {lars_out['err']:.4f}")



print(beta_true)
print(np.mean(result_lasso['beta'], axis=0))
print(np.mean(result_lasso['tz']))
print(np.mean(result_lasso['tau']))
print(np.mean(result_lasso['eta2'], axis=0))




################################################################################################################
# %%
# 2. [DATAGEN_X.R]





import numpy as np
from scipy.stats import norm, uniform, multivariate_normal

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
        x_train = np.column_stack([np.ones(n_train), norm.rvs(size=n_train), uniform.rvs(size=n_train), norm.rvs(size=n_train)])
        x_train = np.column_stack([x_train, np.random.randn(n_train, 5)])
        x_validation = np.column_stack([np.ones(n_validation), norm.rvs(size=n_validation), uniform.rvs(size=n_validation), norm.rvs(size=n_validation)])
        x_validation = np.column_stack([x_validation, np.random.randn(n_validation, 5)])
        x_test = np.column_stack([np.ones(n_test), norm.rvs(size=n_test), uniform.rvs(size=n_test), norm.rvs(size=n_test)])
        x_test = np.column_stack([x_test, np.random.randn(n_test, 5)])
        x = np.vstack([x_train, x_validation, x_test])
        return x
    elif sim_i == 5:
        n_train, n_validation, n_test, p = 20, 20, 400, 30
        beta_true = np.concatenate([np.full(10, 3), np.zeros(10), np.full(10, 3)])
        z1, z2 = norm.rvs(size=1), norm.rvs(size=1)
        x_train = np.vstack([np.concatenate([norm.rvs(loc=z1, scale=0.1, size=10), norm.rvs(size=10), norm.rvs(loc=z2, scale=0.1, size=10)]) for _ in range(n_train)])
        x_validation = np.vstack([np.concatenate([norm.rvs(loc=z1, scale=0.1, size=10), norm.rvs(size=10), norm.rvs(loc=z2, scale=0.1, size=10)]) for _ in range(n_validation)])
        x_test = np.vstack([np.concatenate([norm.rvs(loc=z1, scale=0.1, size=10), norm.rvs(size=10), norm.rvs(loc=z2, scale=0.1, size=10)]) for _ in range(n_test)])
        x = np.vstack([x_train, x_validation, x_test])
        return x
    elif sim_i == 6:
        n_train, n_validation, n_test, p = 20, 20, 400, 15
        beta_true = np.array([-1.2, 1.8, 0, 0, 0, 0, 0.5, 1, 0, 0, 0, 0, 1, 1, 0])
        rho = 0.5
        x_sigma = np.array([[rho**abs(i-j) for j in range(5)] for i in range(5)])
        z_train = multivariate_normal.rvs(mean=np.zeros(5), cov=x_sigma, size=n_train)
        z_validation = multivariate_normal.rvs(mean=np.zeros(5), cov=x_sigma, size=n_validation)
        z_test = multivariate_normal.rvs(mean=np.zeros(5), cov=x_sigma, size=n_test)
        x = np.vstack([z_train, z_validation, z_test])
        return x
    
    rho = 0.5
    x_sigma = np.array([[rho**abs(i-j) for j in range(p)] for i in range(p)])
    x_train = multivariate_normal.rvs(mean=np.zeros(p), cov=x_sigma, size=n_train)
    x_validation = multivariate_normal.rvs(mean=np.zeros(p), cov=x_sigma, size=n_validation)
    x_test = multivariate_normal.rvs(mean=np.zeros(p), cov=x_sigma, size=n_test)
    x = np.vstack([x_train, x_validation, x_test])
    
    return x, beta_true

# Example usage
if __name__ == "__main__":
    x, beta = generate_simulation_data(1)
    print("X shape:", x.shape)
    print("Beta true:", beta)



################################################################################################################
# %%
# 3. [DATAGEN_Y.R]


import numpy as np
from scipy.stats import norm, laplace
from scipy.optimize import root_scalar

def generate_response(x_train, x_validation, x_test, beta_true, sigma_true, sigma_true_1, sigma_true_2, theta, distr):
    np.random.seed(12345)

    if distr == "normal":
        mean = -norm.ppf(theta)
        y_train = x_train @ beta_true + sigma_true * np.random.normal(mean, 1, x_train.shape[0])
        y_validation = x_validation @ beta_true + sigma_true * np.random.normal(mean, 1, x_validation.shape[0])
        y_test = x_test @ beta_true + sigma_true * np.random.normal(mean, 1, x_test.shape[0])

    elif distr == "normalmix":
        def f(x):
            return 0.1 * norm.cdf(-x / sigma_true_1) + 0.9 * norm.cdf(-x / sigma_true_2) - theta

        mean = root_scalar(f, bracket=[-1e4, 1e4]).root
        y_train = x_train @ beta_true + np.where(np.random.rand(x_train.shape[0]) <= 0.1, 
                                                 sigma_true_1, sigma_true_2) * np.random.normal(mean, 1, x_train.shape[0])
        y_validation = x_validation @ beta_true + np.where(np.random.rand(x_validation.shape[0]) <= 0.1, 
                                                           sigma_true_1, sigma_true_2) * np.random.normal(mean, 1, x_validation.shape[0])
        y_test = x_test @ beta_true + np.where(np.random.rand(x_test.shape[0]) <= 0.1, 
                                               sigma_true_1, sigma_true_2) * np.random.normal(mean, 1, x_test.shape[0])

    elif distr == "laplace":
        location = -laplace.ppf(theta)
        y_train = x_train @ beta_true + sigma_true * laplace.rvs(loc=location, scale=1, size=x_train.shape[0])
        y_validation = x_validation @ beta_true + sigma_true * laplace.rvs(loc=location, scale=1, size=x_validation.shape[0])
        y_test = x_test @ beta_true + sigma_true * laplace.rvs(loc=location, scale=1, size=x_test.shape[0])

    elif distr == "laplacemix":
        def f(x):
            return 0.1 * laplace.cdf(-x / sigma_true_1) + 0.9 * laplace.cdf(-x / sigma_true_2) - theta

        location = root_scalar(f, bracket=[-1e4, 1e4]).root
        y_train = x_train @ beta_true + np.where(np.random.rand(x_train.shape[0]) <= 0.1, 
                                                 sigma_true_1, sigma_true_2) * laplace.rvs(loc=location, scale=1, size=x_train.shape[0])

    # Centering the data
    y_train -= np.mean(y_train)
    y_validation -= np.mean(y_validation)
    y_test -= np.mean(y_test)

    return np.concatenate([y_train, y_validation, y_test])



################################################################################################################
# %%
# 4. [SIM.IID.R]

import numpy as np
import joblib  # For loading/saving simulation data
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
import pymc as pm  # Replacing pymc3 with PyMC

def sim_iid(method, n_sim, sim_i, distr, theta, x_train, x_validation, x_test, y_train, y_validation, beta_true):
    results = {}

    if method == "qr_lasso":
        beta_qrl, betasd_qrl, presid_qrl, mad_qrl = [], [], [], []
        for sim in range(1, n_sim + 1):
            print(f"This is simulation number {sim}")
            
            # Load simulation data
            data = joblib.load(f"sim_data/sim_{sim_i}/{distr},theta={theta},sim{sim}.pkl")
            x_train, x_validation, x_test, y_train, y_validation, beta_true = data

            # Preprocess data
            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(np.vstack((x_train, x_validation)))
            y_scaled = np.concatenate((y_train, y_validation)) - np.mean(np.concatenate((y_train, y_validation)))

            # Run QR Lasso model (dummy example call)
            with pm.Model() as model:
                beta = pm.Normal("beta", mu=0, sigma=10, shape=x_scaled.shape[1])
                sigma = pm.HalfCauchy("sigma", beta_true.std())
                mu = pm.math.dot(x_scaled, beta)
                y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_scaled)
                
                trace = pm.sample(2000, return_inferencedata=True)

            # Process results
            beta_est = trace.posterior["beta"].mean(dim=("chain", "draw")).values
            beta_qrl.append(beta_est)
            betasd_qrl.append(trace.posterior["beta"].std(dim=("chain", "draw")).values)
            presid_qrl.append(np.mean((beta_true - beta_est) ** 2))
            mad_qrl.append(np.mean(np.abs(x_test @ (beta_true - beta_est))))

        # Save results
        joblib.dump((beta_qrl, betasd_qrl, presid_qrl, mad_qrl), f"sim_data/sim_{sim_i}/{distr},theta={theta}qrl.pkl")

    elif method == "lasso":
        beta_lasso, lambda_lasso, presid_lasso, mad_lasso = [], [], [], []
        for sim in range(1, n_sim + 1):
            print(f"This is simulation number {sim}")

            # Load data
            data = joblib.load(f"sim_data/sim_{sim_i}/{distr},theta={theta},sim{sim}.pkl")
            x_train, x_validation, x_test, y_train, y_validation, beta_true = data

            # Fit LASSO model (example with sklearn)
            from sklearn.linear_model import LassoCV
            lasso = LassoCV(cv=5).fit(x_train, y_train)

            beta_lasso.append(lasso.coef_)
            lambda_lasso.append(lasso.alpha_)
            presid_lasso.append(np.mean((beta_true - lasso.coef_) ** 2))
            mad_lasso.append(np.mean(np.abs(x_test @ (beta_true - lasso.coef_))))

        joblib.dump((beta_lasso, lambda_lasso, presid_lasso, mad_lasso), f"sim_data/sim_{sim_i}/{distr},theta={theta}lasso.pkl")



################################################################################################################
# %%
# 5. [QRL1_V2.R]

import numpy as np
from scipy.optimize import linprog

def QReg(y, X, maxStep, tau=0.5, error=1e-8, savePath=True, stepSize=100):
    n, p = X.shape
    seqN = np.arange(n)
    seqP = np.arange(p)
    allPredIncluded = False

    # Initialize variables
    Const0 = 1
    Const = np.ones(p)
    
    # Initial conditions
    sortY = np.sort(y)
    nTau = n * tau

    if nTau == int(nTau):  
        Beta0 = sortY[int(nTau) - 1]
    else:
        Beta0 = sortY[int(nTau)]

    Beta = np.zeros(p)
    Residual = y - Beta0

    L = Residual < 0
    R = Residual > 0
    E = Residual == 0

    Alpha = np.where(R, tau, -1 + tau)
    Alpha[E] = tau + np.floor(nTau) - nTau

    ObjVal = tau * np.sum(Residual[R]) - (1 - tau) * np.sum(Residual[L])

    if np.sum(E) != 1 or abs(np.sum(Alpha)) > error:
        print("Numerical error at the beginning")
        return None

    corr = np.dot(Alpha, X) * Const
    maxCorr = np.max(np.abs(corr))
    index = np.abs(corr) == maxCorr

    V = np.zeros(p, dtype=bool)
    V[index] = True
    Sign = np.sign(corr[index])

    Lambda = np.abs(corr[index])
    S = 0

    BetaMatr = [np.hstack([Beta0, Beta])]
    LamTrace = [Lambda]
    STrace = [S]
    ObjValTrace = [ObjVal]

    Step = 1

    while Step <= maxStep and Lambda > error:
        nV = np.sum(V)
        nE = np.sum(E)

        A = np.zeros((nV + 1, nV + 1))
        A[:nE, 0] = 1
        A[:nE, 1:(1 + nV)] = X[E][:, V]
        A[nE, 1:(1 + nV)] = Sign / Const[V]

        b = np.zeros(nE + 1)
        b[nE] = 1

        sol = np.linalg.lstsq(A, b, rcond=None)[0]
        d_beta0 = sol[0]
        d_beta = sol[1:(1 + nV)]

        d_Residual = -(d_beta0 + np.dot(X[:, V], d_beta))

        d_S1 = np.full(n, np.inf)
        valid_index = (L & (d_Residual > 0)) | (R & (d_Residual < 0))
        d_S1[valid_index] = Residual[valid_index] / (-d_Residual[valid_index])

        ds = np.min(d_S1)

        Beta0 += d_beta0 * ds
        Beta[V] += d_beta * ds
        S += ds
        Residual = y - (Beta0 + np.dot(X[:, V], Beta[V]))

        if savePath or Step % stepSize == 0:
            BetaMatr.append(np.hstack([Beta0, Beta]))
            LamTrace.append(Lambda)
            STrace.append(S)
            ObjValTrace.append(ObjVal)

        Step += 1

    return {
        "Beta0": np.array(BetaMatr)[:, 0],
        "Beta": np.array(BetaMatr)[:, 1:],
        "Lambda": np.array(LamTrace),
        "S": np.array(STrace),
        "ObjValue": np.array(ObjValTrace),
        "Step": Step
    }

def predQReg(newY, newX, g):
    resL1, resL2 = [], []
    for i in range(len(g['S'])):
        pred = g['Beta0'][i] + np.dot(newX, g['Beta'][i])
        res = newY - pred
        resL1.append(np.mean(np.abs(res)))
        resL2.append(np.sqrt(np.mean(res ** 2)))
    return {"resL1": resL1, "resL2": resL2}


################################################################################################################
# %%
# 6. [MAIN.R]


import os
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm, laplace
from sklearn.model_selection import KFold
from sklearn.linear_model import LassoCV, ElasticNetCV
import matplotlib.pyplot as plt

# Set working directory
#working_dir = "D:/Files/My Dropbox/QuantReg"
#os.makedirs(working_dir, exist_ok=True)
#os.chdir(working_dir)

# Load external function files
def load_external_functions():
    # Placeholder for loading external functions if needed
    pass

load_external_functions()

# Define number of simulations
n_sim = 50

# ---------------------------------
# 2.1 Data Generation
# ---------------------------------
def generate_data(sim_i):
    np.random.seed(12345)
    if sim_i == 2:
        for distr in ["normal", "normalmix", "laplace", "laplacemix"]:
            for theta in [0.1, 0.3, 0.5]:
                for sim in range(1, n_sim + 1):
                    p = 8
                    rho = 0.5
                    x_sigma = np.array([[rho ** abs(i - j) for j in range(p)] for i in range(p)])

                    x_train = np.random.multivariate_normal(np.zeros(p), x_sigma, size=20)
                    x_validation = np.random.multivariate_normal(np.zeros(p), x_sigma, size=20)
                    x_test = np.random.multivariate_normal(np.zeros(p), x_sigma, size=200)

                    x = np.vstack((x_train, x_validation, x_test))

                    # Generate response variable
                    if distr == "normal":
                        mean = -norm.ppf(theta)
                        y_train = x_train @ np.array([3, 1.5, 0, 0, 2, 0, 0, 0]) + 3 * np.random.normal(mean, 1, 20)
                        y_validation = x_validation @ np.array([3, 1.5, 0, 0, 2, 0, 0, 0]) + 3 * np.random.normal(mean, 1, 20)
                        y_test = x_test @ np.array([3, 1.5, 0, 0, 2, 0, 0, 0]) + 3 * np.random.normal(mean, 1, 200)

                    # Center the response data
                    y_train -= np.mean(y_train)
                    y_validation -= np.mean(y_validation)
                    y_test -= np.mean(y_test)

                    y = np.concatenate((y_train, y_validation, y_test))

                    # Save the generated data
                    save_path = f"sim_data/sim_{sim_i}/{distr}_theta_{theta}_sim_{sim}.pkl"
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    joblib.dump((x_train, x_validation, x_test, x, y_train, y_validation, y_test, y), save_path)

# ---------------------------------
# 2.2 Analysis
# ---------------------------------
def perform_analysis(sim_i):
    beta_true_list = [
        [3, 1.5, 0, 0, 2, 0, 0, 0],
        [0.85] * 8,
        [5, 0, 0, 0, 0, 0, 0, 0],
        "non i.i.d. case",
        [3] * 10 + [0] * 10 + [3] * 10,
        [-1.2, 1.8, 0, 0, 0, 0, 0.5, 1, 0, 0, 0, 0, 1, 1, 0]
    ]
    beta_true = np.array(beta_true_list[sim_i - 1])

    for distr in ["normal", "normalmix", "laplace", "laplacemix"]:
        for theta in [0.1, 0.3, 0.5]:
            for sim in range(1, n_sim + 1):
                data_path = f"sim_data/sim_{sim_i}/{distr}_theta_{theta}_sim_{sim}.pkl"
                x_train, x_validation, x_test, x, y_train, y_validation, y_test, y = joblib.load(data_path)

                scaler = StandardScaler()
                x_scaled = scaler.fit_transform(np.vstack((x_train, x_validation)))
                y_scaled = np.concatenate((y_train, y_validation)) - np.mean(np.concatenate((y_train, y_validation)))

                # Fit Lasso
                lasso = LassoCV(cv=5).fit(x_scaled, y_scaled)
                beta_est = lasso.coef_

                # Save results
                result_path = f"sim_data/sim_{sim_i}/{distr}_theta_{theta}_results.pkl"
                joblib.dump(beta_est, result_path)

# ---------------------------------
# 2.3 Result Processing
# ---------------------------------
def process_results(sim_i):
    for distr in ["normal", "normalmix", "laplace", "laplacemix"]:
        for theta in [0.1, 0.3, 0.5]:
            result_path = f"sim_data/sim_{sim_i}/{distr}_theta_{theta}_results.pkl"
            beta_est = joblib.load(result_path)

            print(f"Simulation {sim_i}, Distribution: {distr}, Theta: {theta}")
            print("Beta Estimates:", beta_est)

# ---------------------------------
# 3. Real Data Processing
# ---------------------------------
def process_real_data():
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LassoCV
    import numpy as np

    # Fetch the California housing dataset
    housing = fetch_california_housing()
    X = housing.data
    y = housing.target  # Median house value in $100,000s

    # We do not log-transform y here, but you could if you want:
    # y = np.log(y)

    kf = KFold(n_splits=10, shuffle=True, random_state=1234)
    for theta in [0.1, 0.3, 0.5]:
        lasso_errors = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Fit Lasso with 5-fold CV
            lasso = LassoCV(cv=5).fit(X_train_scaled, y_train)
            y_pred = lasso.predict(X_test_scaled)

            # Mean absolute error for each split
            lasso_errors.append(np.mean(np.abs(y_test - y_pred)))

        print(f"Theta {theta}, Mean Lasso Error: {np.mean(lasso_errors)}")

# ---------------------------------
# Run All Steps
# ---------------------------------
if __name__ == "__main__":
    print("Running data generation...")
    generate_data(2)
    
    print("Running analysis...")
    perform_analysis(2)
    
    print("Processing results...")
    process_results(2)
    
    print("Processing real data...")
    process_real_data()

    print("All tasks completed.")



################################################################################################################
# %%
# 7. [resultproc.iid.R]


import numpy as np
import pandas as pd

def process_qr_lasso_results(beta_qrl, mad_qrl, method_name="qr.lasso"):
    """
    Emulates the R logic of collecting mean of coefficients, 
    median of mean absolute deviations (MAD), 
    and standard deviation of MAD for a method.
    """

    # 1. Column-wise mean of beta_qrl -> shape (p,)
    beta_mean = beta_qrl.mean(axis=0)  # same as apply(..., 2, mean) in R

    # 2. Median and standard deviation of MAD
    m_mad = np.median(mad_qrl)   # "Mmad" in R
    sd_mad = np.std(mad_qrl)     # "sd.mad.qrl" in R

    # Combine into a single result row
    # e.g., [beta1_mean, beta2_mean, ..., Mmad, sd]
    result_row = np.concatenate([beta_mean, [m_mad, sd_mad]])

    # Put in a DataFrame with 1 row (for 1 method)
    # We'll label columns automatically as:
    # [beta1, beta2, ..., "Mmad", "sd"]
    col_names = [f"beta{i+1}" for i in range(beta_mean.shape[0])] + ["Mmad", "sd"]

    df = pd.DataFrame([result_row], columns=col_names, index=[method_name])
    return df


if __name__ == "__main__":
    # Example: Suppose we have 20 simulations (n_sim=20) and p=5 coefficients
    np.random.seed(123)
    n_sim, p = 20, 5

    # Fake coefficients for each simulation
    beta_qrl = np.random.normal(loc=0, scale=1, size=(n_sim, p))

    # Fake mean absolute deviations for each simulation
    mad_qrl = np.random.exponential(scale=0.5, size=n_sim)

    # Let's pretend we want to process "qr.lasso" results
    df_result = process_qr_lasso_results(beta_qrl, mad_qrl, method_name="qr.lasso")

    # Print the result, similar to R data.frame
    print(df_result.round(4))


################################################################################################################
# %%
# 8. [MAIN.R]
# DOING MAIN AGAIN HERE SO THAT WE CAN CALCULATE MMAD


import numpy as np
import pandas as pd

def process_method_results(beta_matrix, mad_vector, method_name):
    """
    Emulates the row-building in R:
      c(apply(beta_matrix, 2, mean), median(mad_vector), sd(mad_vector))
    and returns a single-row DataFrame with row label = method_name.
    """
    # 1. Column-wise means of coefficients (like apply(..., 2, mean) in R)
    beta_mean = beta_matrix.mean(axis=0)  # shape (p,)

    # 2. Median and std of the MAD vector
    m_mad = np.median(mad_vector)  # "Mmad"
    sd_mad = np.std(mad_vector)

    # Combine into one row
    row_values = np.concatenate([beta_mean, [m_mad, sd_mad]])

    # Column names: [beta1, beta2, ..., Mmad, sd]
    col_names = [f"beta{i+1}" for i in range(beta_mean.shape[0])] + ["Mmad", "sd"]

    # Create a 1-row DataFrame for easy concatenation later
    df = pd.DataFrame([row_values], columns=col_names, index=[method_name])
    return df


if __name__ == "__main__":
    # Suppose we have some methods to process, e.g., "qr.lasso", "qr.enet", "lasso", "enet"
    methods = ["qr.lasso", "qr.enet", "lasso", "enet"]

    # We'll store the final results in a single DataFrame (akin to R's 'temp')
    results_df = pd.DataFrame()

    # This simulates the loop over methods in your R code:
    for method in methods:
        # For demonstration, let's generate random data:
        #  - Suppose n_sim=10, p=5
        np.random.seed(42 + hash(method) % 10000)  # just to get different values for each method
        n_sim, p = 10, 5

        # Fake coefficient matrix: shape (n_sim, p)
        beta_matrix = np.random.normal(loc=0, scale=1, size=(n_sim, p))

        # Fake MAD vector: length n_sim
        mad_vector = np.random.rand(n_sim)

        # Process results for this method
        df_one_method = process_method_results(beta_matrix, mad_vector, method)

        # Concatenate to overall DataFrame (like rbind)
        # If results_df is empty, we just set it to the single row. Otherwise, we append.
        if results_df.empty:
            results_df = df_one_method
        else:
            results_df = pd.concat([results_df, df_one_method], axis=0)

    # Now we have a DataFrame with one row per method, columns: [beta1,...,beta5,"Mmad","sd"]
    print("Final Results:\n")
    print(results_df.round(4))
