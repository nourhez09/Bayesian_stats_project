
################################################################################################################
# %%
# 1. [FUNCTIONS.R]
import numpy as np
import pytensor
import pytensor.tensor as pt
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


print(beta_true)
print(np.mean(result['beta'], axis=0))
print(np.mean(result['tz']))
print(np.mean(result['tau']))
print(np.mean(result['eta2'], axis=0))




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
from pymc3 import Model, sample, find_MAP

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
            result_qrl = qr_lasso(x_scaled, y_scaled, theta)

            # Process results
            beta_est = find_MAP(result_qrl["beta"])
            beta_qrl.append(beta_est)
            betasd_qrl.append(np.std(result_qrl["beta"], axis=0))
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

    # Implement other methods similarly


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
    from sklearn.datasets import load_boston

    boston = load_boston()
    x_boston = boston.data[:, [3, 4, 5, 8, 9, 10, 11, 12]]
    y_boston = np.log(boston.target)

    kf = KFold(n_splits=10, shuffle=True, random_state=1234)
    for theta in [0.1, 0.3, 0.5]:
        lasso_errors = []

        for train_index, test_index in kf.split(x_boston):
            x_train, x_test = x_boston[train_index], x_boston[test_index]
            y_train, y_test = y_boston[train_index], y_boston[test_index]

            scaler = StandardScaler()
            x_train_scaled = scaler.fit_transform(x_train)
            x_test_scaled = scaler.transform(x_test)

            # Fit Lasso
            lasso = LassoCV(cv=5).fit(x_train_scaled, y_train)
            y_pred = lasso.predict(x_test_scaled)
            lasso_errors.append(np.mean(np.abs(y_test - y_pred)))

        print(f"Theta {theta}, Mean Lasso Error: {np.mean(lasso_errors)}")

# ---------------------------------
# Run All Steps
# ---------------------------------
if __name__ == "__main__":
    print("Running data generation...")
    generate_data(2)
    
    print("Running analysis...")
    perform_analysis(1)
    
    print("Processing results...")
    process_results(1)
    
    print("Processing real data...")
    process_real_data()

    print("All tasks completed.")
    y_test = x_test @ beta_true + np.random.normal(0, 1, 50)
    result = qr_lasso(x_test, y_test, theta=0.5, n_sampler=500, n_burn=100, thin=10)
    print("Estimated beta:", np.mean(result['beta'], axis=0))

