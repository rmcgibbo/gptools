# Copyright 2014 Mark Chilenski
# This program is distributed under the terms of the GNU General Purpose License (GPL).
# Refer to http://www.gnu.org/licenses/gpl.txt
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Provides the :py:class:`MaternKernel` class which implements the anisotropic Matern kernel.
"""

from __future__ import division


from .core import ChainRuleKernel, ArbitraryKernel, Kernel
from ..utils import generate_set_partitions, unique_rows
from ._matern import _matern52
from .warping import betacdf_warp

import inspect
import scipy
import scipy.special
import warnings
try:
    import mpmath
except ImportError:
    warnings.warn("Could not import mpmath. Certain functions of the Matern kernel will not function.",
                  ImportWarning)


def matern_function(Xi, Xj, *args):
    r"""Matern covariance function of arbitrary dimension, for use with :py:class:`ArbitraryKernel`.

    The Matern kernel has the following hyperparameters, always referenced in
    the order listed:

    = ===== ====================================
    0 sigma prefactor
    1 nu    order of kernel
    2 l1    length scale for the first dimension
    3 l2    ...and so on for all dimensions
    = ===== ====================================

    The kernel is defined as:

    .. math::

        k_M = \sigma^2 \frac{2^{1-\nu}}{\Gamma(\nu)}
        \left (\sqrt{2\nu \sum_i\left (\frac{\tau_i^2}{l_i^2}\right )}\right )^\nu
        K_\nu\left(\sqrt{2\nu \sum_i\left(\frac{\tau_i^2}{l_i^2}\right)}\right)

    Parameters
    ----------
    Xi, Xj : :py:class:`Array`, :py:class:`mpf`, tuple or scalar float
        Points to evaluate the covariance between. If they are :py:class:`Array`,
        :py:mod:`scipy` functions are used, otherwise :py:mod:`mpmath`
        functions are used.
    *args
        Remaining arguments are the 2+num_dim hyperparameters as defined above.
    """
    num_dim = len(args) - 2
    nu = args[1]

    if isinstance(Xi, scipy.ndarray):
        if isinstance(Xi, scipy.matrix):
            Xi = scipy.asarray(Xi, dtype=float)
            Xj = scipy.asarray(Xj, dtype=float)

        tau = scipy.asarray(Xi - Xj, dtype=float)
        l_mat = scipy.tile(args[-num_dim:], (tau.shape[0], 1))
        r2l2 = scipy.sum((tau / l_mat)**2, axis=1)
        y = scipy.sqrt(2.0 * nu * r2l2)
        k = 2.0**(1 - nu) / scipy.special.gamma(nu) * y**nu * scipy.special.kv(nu, y)
        k[r2l2 == 0] = 1
    else:
        try:
            tau = [xi - xj for xi, xj in zip(Xi, Xj)]
        except TypeError:
            tau = Xi - Xj
        try:
            r2l2 = sum([(t / l)**2 for t, l in zip(tau, args[2:])])
        except TypeError:
            r2l2 = (tau / args[2])**2
        y = mpmath.sqrt(2.0 * nu * r2l2)
        k = 2.0**(1 - nu) / mpmath.gamma(nu) * y**nu * mpmath.besselk(nu, y)
    k *= args[0]**2.0
    return k

class MaternKernelArb(ArbitraryKernel):
    r"""Matern covariance kernel. Supports arbitrary derivatives. Treats order as a hyperparameter.

    This version of the Matern kernel is painfully slow, but uses :py:mod:`mpmath`
    to ensure the derivatives are computed properly, since there may be issues
    with the regular :py:class:`MaternKernel`.

    The Matern kernel has the following hyperparameters, always referenced in
    the order listed:

    = ===== ====================================
    0 sigma prefactor
    1 nu    order of kernel
    2 l1    length scale for the first dimension
    3 l2    ...and so on for all dimensions
    = ===== ====================================

    The kernel is defined as:

    .. math::

        k_M = \sigma^2 \frac{2^{1-\nu}}{\Gamma(\nu)}
        \left (\sqrt{2\nu \sum_i\left (\frac{\tau_i^2}{l_i^2}\right )}\right )^\nu
        K_\nu\left(\sqrt{2\nu \sum_i\left(\frac{\tau_i^2}{l_i^2}\right)}\right)

    Parameters
    ----------
    **kwargs
        All keyword parameters are passed to :py:class:`~gptools.kernel.core.ArbitraryKernel`.
    """
    def __init__(self, **kwargs):
        param_names = [r'\sigma_f', r'\nu'] + ['l_%d' % (i + 1,) for i in range(0, kwargs.get('num_dim', 1))]
        super(MaternKernelArb, self).__init__(matern_function,
                                              num_params=2 + kwargs.get('num_dim', 1),
                                              param_names=param_names,
                                              **kwargs)

    @property
    def nu(self):
        r"""Returns the value of the order :math:`\nu`.
        """
        return self.params[1]


class MaternKernel(ChainRuleKernel):
    r"""Matern covariance kernel. Supports arbitrary derivatives. Treats order as a hyperparameter.

    The Matern kernel has the following hyperparameters, always referenced in
    the order listed:

    = ===== ====================================
    0 sigma prefactor
    1 nu    order of kernel
    2 l1    length scale for the first dimension
    3 l2    ...and so on for all dimensions
    = ===== ====================================

    The kernel is defined as:

    .. math::

        k_M = \sigma^2 \frac{2^{1-\nu}}{\Gamma(\nu)}
        \left (\sqrt{2\nu \sum_i\left (\frac{\tau_i^2}{l_i^2}\right )}\right )^\nu
        K_\nu\left(\sqrt{2\nu \sum_i\left(\frac{\tau_i^2}{l_i^2}\right)}\right)

    Parameters
    ----------
    num_dim : int
        Number of dimensions of the input data. Must be consistent with the `X`
        and `Xstar` values passed to the :py:class:`~gptools.gaussian_process.GaussianProcess`
        you wish to use the covariance kernel with.
    **kwargs
        All keyword parameters are passed to :py:class:`~gptools.kernel.core.ChainRuleKernel`.

    Raises
    ------
    ValueError
        If `num_dim` is not a positive integer or the lengths of the input
        vectors are inconsistent.
    GPArgumentError
        If `fixed_params` is passed but `initial_params` is not.
    """
    def __init__(self, num_dim=1, **kwargs):
        param_names = [r'\sigma_f', r'\nu'] + ['l_%d' % (i + 1,) for i in range(0, num_dim)]
        super(MaternKernel, self).__init__(num_dim=num_dim,
                                           num_params=num_dim + 2,
                                           param_names=param_names,
                                           **kwargs)

    def _compute_k(self, tau):
        r"""Evaluate the kernel directly at the given values of `tau`.

        Parameters
        ----------
        tau : :py:class:`Matrix`, (`M`, `N`)
            `M` inputs with dimension `N`.

        Returns
        -------
        k : :py:class:`Array`, (`M`,)
            :math:`k(\tau)` (less the :math:`\sigma^2` prefactor).
        """
        y, r2l2 = self._compute_y(tau, return_r2l2=True)
        k = 2.0**(1 - self.nu) / scipy.special.gamma(self.nu) * y**self.nu * scipy.special.kv(self.nu, y)
        k[r2l2 == 0] = 1
        return k

    def _compute_y(self, tau, return_r2l2=False):
        r"""Covert tau to :math:`y=\sqrt{2\nu\sum_i(\tau_i^2/l_i^2)}`.

        Parameters
        ----------
        tau : :py:class:`Matrix`, (`M`, `N`)
            `M` inputs with dimension `N`.
        return_r2l2 : bool, optional
            Set to True to return a tuple of (`y`, `r2l2`). Default is False
            (only return `y`).

        Returns
        -------
        y : :py:class:`Array`, (`M`,)
            Inner argument of function.
        r2l2 : :py:class:`Array`, (`M`,)
            Anisotropically scaled distances. Only returned if `return_r2l2` is True.
        """
        r2l2 = self._compute_r2l2(tau)
        y = scipy.sqrt(2.0 * self.nu * r2l2)
        if return_r2l2:
            return (y, r2l2)
        else:
            return y

    def _compute_y_wrapper(self, *args):
        r"""Convert tau to :math:`y=\sqrt{2\nu\sum_i(\tau_i^2/l_i^2)}`.

        Takes `tau` as an argument list for compatibility with :py:func:`mpmath.diff`.

        Parameters
        ----------
        tau[0] : scalar float
            First element of `tau`.
        tau[1] : And so on...

        Returns
        -------
        y : scalar float
            Inner part of Matern kernel at the given `tau`.
        """
        return self._compute_y(scipy.atleast_2d(scipy.asarray(args, dtype=float)))

    def _compute_dk_dy(self, y, n):
        r"""Evaluate the derivative of the outer form of the Matern kernel.

        Uses the general Leibniz rule to compute the n-th derivative of:

        .. math::

            f(y) = \frac{2^{1-\nu}}{\Gamma(\nu)} y^\nu K_\nu(y)

        Notice that this is very poorly-behaved at :math:`x=0`. There, the
        value is approximated using :py:func:`mpmath.diff` with the `singular`
        keyword. This is rather slow, so if you require a fixed value of `nu`
        you may wish to consider implementing the appropriate kernel separately.

        Parameters
        ----------
        y : :py:class:`Array`, (`M`,)
            `M` inputs to evaluate at.
        n : non-negative scalar int.
            Order of derivative to compute.

        Returns
        -------
        dk_dy : :py:class:`Array`, (`M`,)
            Specified derivative at specified locations.
        """
        warnings.warn("The Matern kernel has not been verified for derivatives. Consider using MaternKernelArb.")

        dk_dy = scipy.zeros_like(y, dtype=float)
        non_zero_idxs = (y != 0)
        for k in xrange(0, n + 1):
            dk_dy[non_zero_idxs] += (scipy.special.binom(n, k) *
                                     scipy.special.poch(1 - k + self.nu, k) *
                                     (y[non_zero_idxs])**(-k + self.nu) *
                                     scipy.special.kvp(self.nu, y[non_zero_idxs], n=n-k))

        # Handle the cases at y=0.
        # Compute the appropriate value using mpmath's arbitrary precision
        # arithmetic. This is potentially slow, but seems to behave pretty
        # well. In cases where the value should be infinite, very large
        # (but still finite) floats are returned with the appropriate sign.
        if n >= 2 * self.nu:
            warnings.warn("n >= 2*nu can yield inaccurate results.", RuntimeWarning)

        # Use John Wright's expression for n < 2 * nu:
        if n < 2.0 * self.nu:
            if n % 2 == 1:
                dk_dy[~non_zero_idxs] = 0.0
            else:
                m = n / 2.0
                dk_dy[~non_zero_idxs] = (
                    (-1.0)**m *
                    2.0**(self.nu - 1.0 - n) *
                    scipy.special.gamma(self.nu - m) *
                    scipy.misc.factorial(n) / scipy.misc.factorial(m)
                )
        else:
            # Fall back to mpmath to handle n >= 2 * nu:
            core_expr = lambda x: x**self.nu * mpmath.besselk(self.nu, x)
            deriv = mpmath.chop(mpmath.diff(core_expr, 0, n=n, singular=True, direction=1))
            dk_dy[~non_zero_idxs] = deriv

        dk_dy *= 2.0**(1 - self.nu) / (scipy.special.gamma(self.nu))

        return dk_dy

    def _compute_dy_dtau(self, tau, b, r2l2):
        r"""Evaluate the derivative of the inner argument of the Matern kernel.

        Uses Faa di Bruno's formula to take the derivative of

        .. math::

            y = \sqrt{2 \nu \sum_i(\tau_i^2 / l_i^2)}

        Parameters
        ----------
        tau : :py:class:`Matrix`, (`M`, `N`)
            `M` inputs with dimension `N`.
        b : :py:class:`Array`, (`P`,)
            Block specifying derivatives to be evaluated.
        r2l2 : :py:class:`Array`, (`M`,)
            Precomputed anisotropically scaled distance.

        Returns
        -------
        dy_dtau: :py:class:`Array`, (`M`,)
            Specified derivative at specified locations.
        """
        deriv_partitions = generate_set_partitions(b)
        dy_dtau = scipy.zeros_like(r2l2, dtype=float)
        non_zero_idxs = (r2l2 != 0)
        for p in deriv_partitions:
            dy_dtau[non_zero_idxs] += self._compute_dy_dtau_on_partition(tau[non_zero_idxs], p, r2l2[non_zero_idxs])

        # Case at tau=0 is handled with mpmath for now.
        # TODO: This is painfully slow! Figure out how to do this analytically!
        derivs = scipy.zeros(tau.shape[1], dtype=int)
        for d in b:
            derivs[d] += 1
        dy_dtau[~non_zero_idxs] = mpmath.chop(
            mpmath.diff(
                self._compute_y_wrapper,
                scipy.zeros(tau.shape[1], dtype=float),
                n=derivs,
                singular=True,
                direction=1
            )
        )
        return dy_dtau

    def _compute_dy_dtau_on_partition(self, tau, p, r2l2):
        """Evaluate the term inside the sum of Faa di Bruno's formula for the given partition.

        Parameters
        ----------
        tau : :py:class:`Matrix`, (`M`, `N`)
            `M` inputs with dimension `N`.
        p : list of :py:class:`Array`
            Each element is a block of the partition representing the derivative
            orders to use.
        r2l2 : :py:class:`Array`, (`M`,)
            Precomputed anisotropically scaled distance.

        Returns
        -------
        dy_dtau : :py:class:`Array`, (`M`,)
            The specified derivatives over the given partition at the specified
            locations.
        """
        n = len(p)
        dy_dtau = scipy.zeros_like(r2l2)
        dy_dtau = (scipy.sqrt(2.0 * self.nu) *
                   scipy.special.poch(1 - n + 0.5, n) *
                   (r2l2)**(-n + 0.5))
        for b in p:
            dy_dtau *= self._compute_dT_dtau(tau, b)

        return dy_dtau

    def _compute_dT_dtau(self, tau, b):
        r"""Evaluate the derivative of the :math:`\tau^2` sum term.

        Parameters
        ----------
            tau : :py:class:`Matrix`, (`M`, `N`)
                `M` inputs with dimension `N`.
            b : :py:class:`Array`, (`P`,)
                Block specifying derivatives to be evaluated.

        Returns
        -------
        dT_dtau : :py:class:`Array`, (`M`,)
            Specified derivative at specified locations.
        """
        unique_d = scipy.unique(b)
        # Derivatives of order 3 and up are zero, mixed derivatives are zero.
        if len(b) >= 3 or len(unique_d) > 1:
            return scipy.zeros(tau.shape[0])
        else:
            tau_idx = unique_d[0]
            if len(b) == 1:
                return 2.0 * tau[:, tau_idx] / (self.params[2 + tau_idx])**2.0
            else:
                # len(b) == 2 is the only other possibility here because of
                # the first test.
                return 2.0 / (self.params[2 + tau_idx])**2.0 * scipy.ones(tau.shape[0])

    @property
    def nu(self):
        r"""Returns the value of the order :math:`\nu`.
        """
        return self.params[1]


class Matern52Kernel(Kernel):
    r"""Matern 5/2 covariance kernel. Supports only 0th and 1st derivatives
    and is fixed at nu=5/2. Because of these limitations, it is quite a bit
    faster than the more general Matern kernels.

    The Matern52 kernel has the following hyperparameters, always referenced in
    the order listed:

    = ===== ====================================
    0 sigma prefactor
    1 l1    length scale for the first dimension
    2 l2    ...and so on for all dimensions
    = ===== ====================================

    The kernel is defined as:

    .. math::

        k_M(x, x') = \sigma^2 \left(1 + \sqrt{5r^2} + \frac{5}{3}r^2\right) \exp(-\sqrt{5r^2}) \\
        r^2 = \sum_{d=1}^D \frac{(x_d - x'_d)^2}{l_d^2}

    Parameters
    ----------
    num_dim : int
        Number of dimensions of the input data. Must be consistent with the `X`
        and `Xstar` values passed to the :py:class:`~gptools.gaussian_process.GaussianProcess`
        you wish to use the covariance kernel with.
    **kwargs
        All keyword parameters are passed to :py:class:`~gptools.kernel.core.Kernel`.

    Raises
    ------
    ValueError
        If `num_dim` is not a positive integer or the lengths of the input
        vectors are inconsistent.
    GPArgumentError
        If `fixed_params` is passed but `initial_params` is not.
    """
    def __init__(self, num_dim=1, **kwargs):
        param_names = [r'\sigma_f'] + ['l_%d' % (i + 1,) for i in range(0, num_dim)]
        super(Matern52Kernel, self).__init__(num_dim=num_dim,
                                             num_params=num_dim + 1,
                                             param_names=param_names,
                                             **kwargs)
    def __call__(self, Xi, Xj, ni, nj, hyper_deriv=None, symmetric=False):
        """Evaluate the covariance between points `Xi` and `Xj` with derivative order `ni`, `nj`.

        Parameters
        ----------
        Xi : :py:class:`Matrix` or other Array-like, (`M`, `N`)
            `M` inputs with dimension `N`.
        Xj : :py:class:`Matrix` or other Array-like, (`M`, `N`)
            `M` inputs with dimension `N`.
        ni : :py:class:`Matrix` or other Array-like, (`M`, `N`)
            `M` derivative orders for set `i`.
        nj : :py:class:`Matrix` or other Array-like, (`M`, `N`)
            `M` derivative orders for set `j`.
        hyper_deriv : Non-negative int or None, optional
            The index of the hyperparameter to compute the first derivative
            with respect to. If None, no derivatives are taken. Hyperparameter
            derivatives are not supported at this point. Default is None.
        symmetric : bool
            Whether or not the input `Xi`, `Xj` are from a symmetric matrix.
            Default is False.

        Returns
        -------
        Kij : :py:class:`Array`, (`M`,)
            Covariances for each of the `M` `Xi`, `Xj` pairs.

        Raises
        ------
        NotImplementedError
            If the `hyper_deriv` keyword is not None.
        """
        if hyper_deriv is not None:
            raise NotImplementedError("Hyperparameter derivatives have not been implemented!")
        if scipy.any(scipy.sum(ni, axis=1) > 1) or scipy.any(scipy.sum(nj, axis=1) > 1):
            raise NotImplementedError("Matern52Kernel only supports 0th and 1st order derivatives")

        Xi = scipy.asarray(Xi, dtype=scipy.float64)
        Xj = scipy.asarray(Xj, dtype=scipy.float64)
        ni = scipy.array(ni, dtype=scipy.int32)
        nj = scipy.array(nj, dtype=scipy.int32)
        var = scipy.square(self.params[-self.num_dim:])

        value = _matern52(Xi, Xj, ni, nj, var)
        return self.params[0]**2 * value


class WarpedMatern52Kernel(Kernel):
    r"""Abstract base class for Matern52Kernel with a warping function applied
    to the coordinates

    The kernel is defined as:

    .. math::

        k_M(x, x') = \sigma^2 \left(1 + \sqrt{5r^2} + \frac{5}{3}r^2\right) \exp(-\sqrt{5r^2}) \\
        r^2 = \sum_{d=1}^D \frac{(w_d(x_d) - w_d(x'_d))^2}{l_d^2}

    where :math:`w_d (x)` is the warping function for the d-th dimension, which
    must be defined by a concrete subclass.

    Parameters
    ----------
    num_dim : int
        Number of dimensions of the input data. Must be consistent with the `X`
        and `Xstar` values passed to the :py:class:`~gptools.gaussian_process.GaussianProcess`
        you wish to use the covariance kernel with.

    length_scale_param_indices : array-like of int, shape=(length num_dim,)
        To evaluate the Matern function, this base class needs to know which
        of its hyperparameters refer to the length scales of the Matern kernel,
        as other hyperparameters might be used for the warping portion. For this
        reason, subclasses must pass in this list of indices into self.params
        giving the length scale hyperparameters. E.g. ::

            length_scales = self.params[self.length_scale_param_indices]

    **kwargs
        All keyword parameters are passed to :py:class:`~gptools.kernel.core.ChainRuleKernel`.
    """
    def __init__(self, num_dim, length_scale_param_indices, **kwargs):
        if not len(length_scale_param_indices) == num_dim:
            raise ValueError('length_scale_param_indices must be of length num_dim')
        self.length_scale_param_indices = scipy.asarray(length_scale_param_indices)

        super(WarpedMatern52Kernel, self).__init__(num_dim=num_dim, **kwargs)

    def __call__(self, Xi, Xj, ni, nj, hyper_deriv=None, symmetric=False):
        if hyper_deriv is not None:
            raise NotImplementedError("Hyperparameter derivatives have not been implemented!")
        if scipy.any(scipy.sum(ni, axis=1) > 1) or scipy.any(scipy.sum(nj, axis=1) > 1):
            raise NotImplementedError("WarpedMatern52Kernel only supports 0th and 1st order derivatives")

        var = scipy.square(self.params[self.length_scale_param_indices], dtype=float)

        n_combined = scipy.asarray(scipy.hstack((ni, nj)), dtype=int)
        n_combined_unique = unique_rows(n_combined)

        k = scipy.zeros(Xi.shape[0], dtype=float)
        wXi = scipy.zeros(Xi.shape, dtype=float)
        wXj = scipy.zeros(Xi.shape, dtype=float)
        for d in range(self.num_dim):
            wXi[:, d] = self._warp_func(Xi[:, d], 0, d)
            wXj[:, d] = self._warp_func(Xj[:, d], 0, d)

        for n_combined_state in n_combined_unique:
            idxs = (n_combined == n_combined_state).all(axis=1)
            ni_idxs = scipy.asarray(ni[idxs], order='c', dtype=scipy.int32)
            nj_idxs = scipy.asarray(nj[idxs], order='c', dtype=scipy.int32)
            wXi_idxs = scipy.asarray(wXi[idxs], order='c', dtype=float)
            wXj_idxs = scipy.asarray(wXj[idxs], order='c', dtype=float)
            k[idxs] = _matern52(wXi_idxs, wXj_idxs, ni_idxs, nj_idxs, var)
            if scipy.sum(n_combined_state) == 0:
                continue

            ni_state = n_combined_state[:self.num_dim]
            nj_state = n_combined_state[self.num_dim:]
            for d in range(self.num_dim):
                if ni_state[d] > 0:
                    k[idxs] *= self._warp_func(Xi[idxs, d], ni_idxs[:, d], d)
                if nj_state[d] > 0:
                    k[idxs] *= self._warp_func(Xj[idxs, d], nj_idxs[:, d], d)

        return self.params[0]**2 * k

    def _warp_func(self, x, n, d):
        """The coordinate-warping function.

        Parameters
        ---------
        x : array-like, dtype=float, shape=(M,)
            This should be the `d`th column of Xi or Xj.
        n : int
            Derivative order for these observations
        d : int
            The index of the dimension that's being warped. Different dimensions
            can be warped differently.
        """
        raise NotImplementedError()


class BetaCDFWarpedMatern52Kernel(WarpedMatern52Kernel):
    r"""Non-stationary BetaCDF-warped Matern 5/2

    The BetaCDFWarpedMatern52Kernel kernel has the following hyperparameters,
    always referenced in the order listed:

    = =====  ====================================
    0 sigma  prefactor
    1 l1     length scale for the first dimension
    2 alpha1 alpha warping parameter for first dimension
    3 beta1  beta warping parameter for first dimension
    4 l2    ...and so on for all dimensions
    = ===== ===================================

    The kernel is defined as:

    .. math::

        k_M(x, x') = \sigma^2 \left(1 + \sqrt{5r^2} + \frac{5}{3}r^2\right) \exp(-\sqrt{5r^2}) \\
        r^2 = \sum_{d=1}^D \frac{(w_d(x_d) - w_d(x'_d))^2}{l_d^2}

    where ..math::

        w_d(x) = BetaCDF(x, \alpha_d, \beta_d)

    Some properties and applications of this kernel are described in Snoek et
    al. [1].

    Parameters
    ----------
    num_dim : int
        Number of dimensions of the input data. Must be consistent with the `X`
        and `Xstar` values passed to the :py:class:`~gptools.gaussian_process.GaussianProcess`
        you wish to use the covariance kernel with.
    **kwargs
        All keyword parameters are passed to :py:class:`~gptools.kernel.core.Kernel`.

    Raises
    ------
    ValueError
        If `num_dim` is not a positive integer or the lengths of the input
        vectors are inconsistent.
    GPArgumentError
        If `fixed_params` is passed but `initial_params` is not.

    References
    ----------
    .. [1] J. Snoek, K. Swersky, R. Zemel, R. P. Adams, "Input Warping for
       Bayesian Optimization of Non-stationary Functions" ICML (2014)
    """
    def __init__(self, num_dim=1, **kwargs):
        param_names = [r'\sigma_f']
        for i in range(num_dim):
            param_names.extend([
                'l_%d' % (i+1,),
                'alpha_%d' % (i+1,),
                'beta_%d' % (i+1,)
            ])
        length_scale_param_indices = [i for i, name in enumerate(param_names)
            if name.startswith('l_')]

        super(BetaCDFWarpedMatern52Kernel, self).__init__(
            num_dim=num_dim,
            length_scale_param_indices=length_scale_param_indices,
            num_params=len(param_names),
            param_names=param_names, **kwargs)

    def _warp_func(self, x, n, d):
        alpha = self.params[2 + 3*d]
        beta = self.params[3 + 3*d]
        return betacdf_warp(x, n, alpha, beta)
