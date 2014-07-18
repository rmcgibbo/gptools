import numpy as np
from scipy.optimize import check_grad
from gptools.kernel.warping import betacdf_warp
from numpy.testing import assert_raises
random = np.random.RandomState(0)

def test_1():
    # check that things run with vector inputs
    x = random.uniform(size=10)
    alpha = random.lognormal()
    beta = random.lognormal()
    zeros = np.zeros(10)
    ones = np.zeros(10)

    result = betacdf_warp(x, zeros, alpha, beta)
    assert result.shape == (10,)

    derivs = betacdf_warp(x, ones, alpha, beta)
    assert derivs.shape == (10,)


def test_2():
    # check the derivative calculation, and also that
    # everything works with scalar inputs
    alpha = 0.5
    beta = 2

    def func(x):
        return betacdf_warp(x, 0, alpha, beta)
    def grad(x):
        return betacdf_warp(x, 1, alpha, beta)

    for i in range(10):
        assert check_grad(func, grad, x0=random.uniform(size=1)) < 1e-6


def test_3():
    # check that the right errors get thrown

    with assert_raises(ValueError):
        betacdf_warp(10, 0, 0.5, 0.5)

    with assert_raises(ValueError):
        betacdf_warp(0.5, 0, -1, 0.5)

    with assert_raises(ValueError):
        betacdf_warp(0.5, 0, 0.5, -1)

    with assert_raises(NotImplementedError):
        betacdf_warp(0.5, 5, 0.5, 0.5)


def test_4():
    # check that it works when X is more than 1 dimensional
    X = random.uniform(size=(5,2))
    alpha = [1, 2]
    beta = [3, 1]
    result = betacdf_warp(X, 0, alpha, beta)

    np.testing.assert_array_almost_equal(
        betacdf_warp(X[:,0], 0, alpha[0], beta[0]),
        result[:,0])

    np.testing.assert_array_almost_equal(
        betacdf_warp(X[:,1], 0, alpha[1], beta[1]),
        result[:,1])


from gptools.kernel import Matern52KernelBetaCDF, Matern52Kernel

def test_5():
    # set alpha=beta=1 in Matern52KernelBetaCDF, and check that
    # its the same as Matern52Kernel
    x = random.uniform(size=(10, 1))
    n = np.zeros((10, 1))

    k1 = Matern52KernelBetaCDF(num_dim=1)
    k1.params = np.array([1, 1, 1, 1], dtype=float)
    k2 = Matern52Kernel(num_dim=1)
    k2.params = np.array([1, 1], dtype=float)

    np.testing.assert_array_equal(
        k1(x,x,n,n), k2(x,x,n,n))


def test_6():
    # finite-difference verification of Matern52KernelBetaCDF
    # derivatives in 1D
    k1 = Matern52KernelBetaCDF(num_dim=1)
    k1.params = np.array([1, 2, 3, 4], dtype=float)
    y = np.array([[0.2]])

    def func(x):
        nx = np.zeros((1,1))
        ny = np.zeros((1,1))
        return k1(x, y, nx, ny)

    def grad(x):
        nx = np.ones((1,1))
        ny = np.zeros((1,1))
        return k1(x, y, nx, ny)

    for i in range(10):
        x0 = random.uniform(size=(1,1))
        assert check_grad(func, grad, x0) < 1e-7


def test_7():
    # finite-difference verification of Matern52KernelBetaCDF
    # derivatives in 2D
    k1 = Matern52KernelBetaCDF(num_dim=2)
    k1.params = np.array([1,   2, 3, 4,   0.2, 0.3, 0.4], dtype=float)
    y = np.array([[0.2, 0.3]])

    def func(x):
        nx = np.zeros((1,2))
        ny = np.zeros((1,2))
        return k1(x, y, nx, ny)

    def grad(x, which=0):
        nx = np.zeros((1,2))
        nx[0, which] = 1
        ny = np.zeros((1,2))
        return k1(x, y, nx, ny)

    h = 1e-7
    x = random.uniform(size=(1,2))
    x_prime_0 = x + [h, 0]
    x_prime_1 = x + [0, h]

    fd_0 = (func(x_prime_0) - func(x)) / h
    fd_1 = (func(x_prime_1) - func(x)) / h

    grad_0 = grad(x, which=0)
    grad_1 = grad(x, which=1)

    np.testing.assert_almost_equal(fd_0, grad_0, decimal=6)
    np.testing.assert_almost_equal(fd_1, grad_1, decimal=6)
