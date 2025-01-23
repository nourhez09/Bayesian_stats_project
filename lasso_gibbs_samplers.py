import numpy as np
from scipy.stats import geninvgauss, norm, gamma, expon, laplace, invgauss
from numpy.linalg import solve
from metrics import calculate_mmad


import numpy as np
from scipy.stats import gamma, expon, laplace, norm
from scipy.stats import geninvgauss

def gibbs_sampler(X, y, theta, n_iter=1000, a=0, b=0, c=0, d=0):
    """
    Gibbs sampler for Bayesian Lasso Quantile Regression.

    Parameters:
        X : numpy.ndarray
            Design matrix (n x p).
        y : numpy.ndarray
            Response vector (n x 1).
        theta : float
            Quantile level (e.g., 0.5 for median).
        n_iter : int
            Number of iterations for the sampler.

    Returns:
        Dictionary containing posterior samples for all parameters.
    """

    # Dimensions
    n, p = X.shape

    # Quantile-related constants
    ksi_1 = (1 - 2 * theta) / (theta * (1 - theta))
    ksi_2 = np.sqrt(2 / (theta * (1 - theta)))

    # Initialize priors and posterior storage
    samples = {
        "beta": np.zeros((n_iter, p)),
        "tau": np.zeros(n_iter),
        "eta2": np.zeros(n_iter),
        "v": np.zeros((n_iter, n)),
        "s": np.zeros((n_iter, p))
    }

    # Initial values
    samples["v"][0, :] = expon.rvs(scale=1, size=n)  # Exponential prior for v
    samples["s"][0, :] = gamma.rvs(a=1, scale=1, size=p)  # Gamma prior for s
    samples["tau"][0] = gamma.rvs(a=1, scale=1)  # Gamma prior for tau
    samples["eta2"][0] = gamma.rvs(a=1, scale=1)  # Gamma prior for eta^2
    samples["beta"][0, :] = laplace.rvs(loc=0, scale=1, size=p)  # Laplace prior for beta

    # Gibbs sampling
    for it in range(1, n_iter):
        # Previous iteration values
        beta = samples["beta"][it - 1, :]
        tau = samples["tau"][it - 1]
        eta2 = samples["eta2"][it - 1]
        v = samples["v"][it - 1, :]
        s = samples["s"][it - 1, :]


        # Sample v_i
        for i in range(n):
            chi_v = tau * (y[i] - np.dot(X[i, :], beta))** 2 / ksi_2**2
            psi_v = (2*tau + tau*ksi_1**2/(ksi_2**2))
            v[i] = geninvgauss.rvs(0.5, chi_v, psi_v)
    
        # Sample s_k
        for k in range(p):
            lambda_param=0.5
            chi_sk = beta[k] ** 2
            psi_sk = eta2
            s[k] = geninvgauss.rvs(lambda_param, chi_sk, psi_sk)

        # Sample beta_k
        for k in range(p):
            var_k = 1 / (tau / ksi_2**2 * np.sum(X[:, k] ** 2 / v) + tau / s[k])
            c_k = y - ksi_1 * v - np.dot(X, beta) + beta[k] * X[:, k]
            mu_k = var_k * tau / ksi_2**2 * np.sum(X[:, k] * c_k / v)
            beta[k] = norm.rvs(loc=mu_k, scale=np.sqrt(var_k))


                # Sample tau
        shape_tau = a + (3 * n) / 2
        rate_tau = 0.5 * np.sum(
            (y - X @ beta - ksi_1 * v) ** 2 / (ksi_2**2 * v)
            + v
        ) + b
        tau = gamma.rvs(shape_tau, scale=1 / rate_tau)

        # Sample eta^2
        shape_eta2 = c+p + 1
        rate_eta2 = 0.5 * np.sum(s) + d
        eta2 = gamma.rvs(shape_eta2, scale=1 / rate_eta2)

        # Store samples
        samples["beta"][it, :] = beta
        samples["tau"][it] = tau
        samples["eta2"][it] = eta2
        samples["v"][it, :] = v
        samples["s"][it, :] = s

    return samples



# Function to create the covariance matrix Sigma
def create_covariance_matrix(p):
    sigma = np.fromiter((0.5**abs(i-j) for i in range(p) for j in range(p)), dtype=float)
    return sigma.reshape(p, p)

# Example usage
if __name__ == "__main__":
    # Simulate some data
    # np.random.seed(42)

    from scipy.stats import multivariate_normal     

    n, p = 100, 9

        # Given standard deviation
    sigma_X = create_covariance_matrix(p) # example standard deviation
    sigma=3
    # Desired quantile
    theta = 0.1  # example quantile, e.g., the 75th quantile
    # Compute the value of mu that makes the theta-th quantile equal to 0
    mu = -sigma * norm.ppf(theta)
    # X = np.random.randn(n, p)
    X = np.random.multivariate_normal(mean=np.zeros(p), cov=sigma_X, size=n)
    u = np.random.normal(loc=mu, scale=sigma, size=n)
    # true_beta = np.array([1.5, -2.0, 0.0, 0.0, 0.5])
    true_beta = np.array([3.0, 1.0, 5.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    y = X @ true_beta + u

    # Run Gibbs sampler
    iterations=[]
    for i in range(2):
        result = gibbs_sampler(X, y,theta, n_iter=1000)
        iterations.append(result["beta"][-1])
    print('mmad is ', calculate_mmad(iterations, X, y))








# # Number of rows and columns (dimension of X)
# n = 5  # for example, adjust as needed

# # Generate the covariance matrix Sigma
# Sigma = create_covariance_matrix(n)

# # Generate X where each row is drawn from N(0, Sigma)
# X = np.random.multivariate_normal(mean=np.zeros(n), cov=Sigma, size=n)




