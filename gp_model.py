# gp_model.py
import numpy as np
from scipy.optimize import minimize


class RBFKernel:
    """
    Radial Basis Function (RBF) kernel with optional white noise term.

    k(x, x') = sigma_f^2 * exp(-0.5 * ||(x - x') / ℓ||^2) + sigma_w^2 * I
    """

    def __init__(self, lengthscales, sigma_f2=1.0, sigma_w2=1e-3):
        # Immutable-ish scalars + numpy array (mutable) to satisfy requirements
        self.lengthscales = np.asarray(lengthscales, dtype=float)
        self.sigma_f2 = float(sigma_f2)
        self.sigma_w2 = float(sigma_w2)

    def __call__(self, X1, X2, add_noise=False):
        """Compute kernel matrix between X1 and X2."""
        # kernel = RBFKernel(lengthscales=[1.0, 1.0])
        # K = kernel(X1, X2)
        # Even though kernel is an object, you can “call” it like a function.
        # That is the main goal of __call__
        X1 = np.asarray(X1, dtype=float)
        X2 = np.asarray(X2, dtype=float)

        # Scale each input dimension by its corresponding lengthscale.
        # This ensures the RBF kernel handles different smoothness per feature.
        # Equivalent to applying Λ⁻¹ in the exponent of the kernel:
        #     k(x, x') = σ_f² * exp(-0.5 * (x - x')ᵀ Λ⁻¹ (x - x'))
        # Uses broadcasting: X.shape = (n, d), lengthscales.shape = (d,)
        X1_scaled = X1 / self.lengthscales
        X2_scaled = X2 / self.lengthscales

        # Efficient pairwise squared distance matrix:
        # sqdist[i, j] = ||X1[i] - X2[j]||²
        # Using broadcasting:
        # - [:, None] makes a column vector (n, 1)
        # - [None, :] makes a row vector (1, m)
        sqdist = (
            np.sum(X1_scaled ** 2, axis=1)[:, None]
            + np.sum(X2_scaled ** 2, axis=1)[None, :]
            - 2 * np.dot(X1_scaled, X2_scaled.T)
        )
        K = self.sigma_f2 * np.exp(-0.5 * sqdist)

        # Optional white-noise term
        if add_noise and X1.shape[0] == X2.shape[0]:
            K += (self.sigma_w2 + 1e-6) * np.eye(len(X1))
        return K
    

    # ----- Operator overloading (Requirement Part 2) -----
    def __mul__(self, other):
        """
        Overload * to scale the kernel variance.
        Example: 2.0 * kernel  ->  kernel with 2x sigma_f^2
        """
        if isinstance(other, (int, float)):
            new_sigma_f2 = self.sigma_f2 * float(other)
            return RBFKernel(self.lengthscales, new_sigma_f2, self.sigma_w2)
        raise TypeError("RBFKernel can only be multiplied by a scalar.")

    __rmul__ = __mul__  # allow scalar * kernel as well


class GaussianProcess:
    """
    Simple Gaussian Process Regression model using an RBF kernel.
    """

    def __init__(self, kernel: RBFKernel):
        self.kernel = kernel
        self.X_train = None
        self.y_train = None
        self.L = None  # Cholesky factor
        self.alpha = None
        self.log_marginal_likelihood_ = None

    # ---------- Internal helper functions ----------
    def _compute_K(self, X):
        """Compute training covariance matrix K."""
        return self.kernel(X, X, add_noise=True)

    def _negative_log_likelihood(self, log_params, X, y):
        """
        Negative log marginal likelihood as a function of log hyper-parameters.
        log_params = [log(sigma_f2), log(sigma_w2), log(l1), log(l2)]
        """
        sigma_f2 = np.exp(log_params[0])
        sigma_w2 = np.exp(log_params[1])
        lengthscales = np.exp(log_params[2:])

        # Update kernel hyperparameters temporarily
        self.kernel.sigma_f2 = sigma_f2
        self.kernel.sigma_w2 = sigma_w2
        self.kernel.lengthscales = lengthscales

        K = self._compute_K(X)

        #Cholesky decomposition takes a symmetric positive-definite matrix K and factorizes it into:
        # K=L⋅L⊤
        # K is your covariance (kernel) matrix
        # L is a lower triangular matrix (same size as 𝐾)
        # 𝐿⊤ is its transpose (upper triangular)

        # Example:
        # K = np.array([[4, 2],
        #         [2, 3]])
        # L = np.linalg.cholesky(K)
        # print(L)

        # Output:
        # [[2.         0.        ]
        #  [1.         1.41421356]]

        # ✅ Check:
        # L @ L.T # @ is just the matrix multiplication operator in Python.
        # array([[4., 2.],
        #        [2., 3.]])  ✅ == K
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return np.inf  # Non-PD -> bad parameters

        # Compute α = K⁻¹ y using Cholesky factorization:
        # First solve L z = y  → z = L⁻¹ y
        # Then solve Lᵀ α = z → α = (Lᵀ)⁻¹ z = K⁻¹ y
        # This avoids directly inverting K, making it faster and more stable.
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        log_det_K = 2.0 * np.sum(np.log(np.diag(L)))
        
        # So, where's the negative?
        # The answer is: you don't put a negative inside the equation — you return the positive value from the function, 
        # and then the optimizer minimizes i
        nll = 0.5 * y @ alpha + 0.5 * log_det_K + 0.5 * len(y) * np.log(2 * np.pi) # @ is just the matrix multiplication operator in Python.
        return float(nll)  # immutable scalar (Requirement: immutable type)


    # ---------- Public API ----------
    def fit(self, X, y):
        """
        Train GP by optimizing kernel hyperparameters.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        self.X_train = X
        self.y_train = y

        # Initial hyper-parameters (log-space)
        initial_log_params = np.log(
            [self.kernel.sigma_f2, self.kernel.sigma_w2, *self.kernel.lengthscales]
        )

        bounds = [
            (np.log(1e-2), np.log(10.0)),   # sigma_f^2
            (np.log(1e-2), np.log(1.0)),    # sigma_w^2
            (np.log(0.01), np.log(10.0)),   # l1
            (np.log(0.01), np.log(10.0)),   # l2
        ]

        res = minimize(
            self._negative_log_likelihood,
            initial_log_params,
            args=(X, y),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 100, "disp": True},
        )

        # Store optimized hyper-parameters
        opt_params = np.exp(res.x)
        self.kernel.sigma_f2 = opt_params[0]
        self.kernel.sigma_w2 = opt_params[1]
        self.kernel.lengthscales = opt_params[2:]

        # Precompute Cholesky and alpha for predictions
        K = self._compute_K(self.X_train)
        self.L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y_train))
        self.log_marginal_likelihood_ = -float(res.fun)

        return self

    def predict(self, X_test, return_std=False):
        """
        Predict mean (and optionally standard deviation) at test points.
        """
        X_test = np.asarray(X_test, dtype=float)

        K_s = self.kernel(self.X_train, X_test, add_noise=False)
        K_ss = self.kernel(X_test, X_test, add_noise=False)

        # Predictive mean
        mean = K_s.T @ self.alpha

        # Predictive covariance
        v = np.linalg.solve(self.L, K_s)
        cov = K_ss - v.T @ v
        var = np.clip(np.diag(cov), a_min=1e-10, a_max=None)

        if return_std:
            return mean, np.sqrt(var)
        return mean
