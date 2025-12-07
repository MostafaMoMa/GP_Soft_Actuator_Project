# test_gp_model.py
import numpy as np
from gp_model import RBFKernel, GaussianProcess


def test_rbf_kernel_symmetry():
    X = np.array([[0.0, 1.0], [1.0, 2.0]])
    kernel = RBFKernel(lengthscales=[1.0, 1.0], sigma_f2=2.0, sigma_w2=0.1)

    K = kernel(X, X, add_noise=True)

    # Symmetry
    assert np.allclose(K, K.T, atol=1e-8)

    # Positive values on diagonal
    assert np.all(np.diag(K) > 0.0)


def test_negative_log_likelihood_finite():
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    y = np.array([0.0, 1.0, 2.0])

    kernel = RBFKernel(lengthscales=[1.0, 1.0], sigma_f2=1.0, sigma_w2=1e-3)
    gp = GaussianProcess(kernel)

    # Fit should result in finite log marginal likelihood
    gp.fit(X, y)
    assert np.isfinite(gp.log_marginal_likelihood_)


if __name__ == "__main__":
    import pytest
    # __file__ is a string containing the path of the current Python file that is being executed.
    pytest.main(["-v", __file__])

