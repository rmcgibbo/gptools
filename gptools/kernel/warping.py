# Copyright 2014 Mark Chilenski
# Author: Mark Chilenski
# Contributors: Robert McGibbon
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

"""Provides warping functions for building non-stationary kernels
"""

try:
    import mpmath
except ImportError:
    import warnings
    warnings.warn("Could not import mpmath. Arbitrary warping functions for Gibbs kernels will not function.",
                  ImportWarning)

import numbers
import scipy
import scipy.special

__all__ = ['cubic_bucket_warp', 'double_tanh_warp', 'gauss_warp_arb', 'quintic_bucket_warp',
           'tanh_warp', 'tanh_warp_arb']


def tanh_warp(x, n, l1, l2, lw, x0):
    r"""Implements a tanh warping function and its derivative.

    .. math::

        l = \frac{l_1 + l_2}{2} - \frac{l_1 - l_2}{2}\tanh\frac{x-x_0}{l_w}

    Parameters
    ----------
    x : float or array of float
        Locations to evaluate the function at.
    n : int
        Derivative order to take. Used for ALL of the points.
    l1 : positive float
        Left saturation value.
    l2 : positive float
        Right saturation value.
    lw : positive float
        Transition width.
    x0 : float
        Transition location.

    Returns
    -------
    l : float or array
        Warped length scale at the given locations.

    Raises
    ------
    NotImplementedError
        If `n` > 1.
    """
    if n == 0:
        return (l1 + l2) / 2.0 - (l1 - l2) / 2.0 * scipy.tanh((x - x0) / lw)
    elif n == 1:
        return -(l1 - l2) / (2.0 * lw) * (scipy.cosh((x - x0) / lw))**(-2.0)
    else:
        raise NotImplementedError("Only derivatives up to order 1 are supported!")


def tanh_warp_arb(X, l1, l2, lw, x0):
    r"""Warps the `X` coordinate with the tanh model

    .. math::

        l = \frac{l_1 + l_2}{2} - \frac{l_1 - l_2}{2}\tanh\frac{x-x_0}{l_w}

    Parameters
    ----------
    X : :py:class:`Array`, (`M`,) or scalar float
        `M` locations to evaluate length scale at.
    l1 : positive float
        Small-`X` saturation value of the length scale.
    l2 : positive float
        Large-`X` saturation value of the length scale.
    lw : positive float
        Length scale of the transition between the two length scales.
    x0 : float
        Location of the center of the transition between the two length scales.

    Returns
    -------
    l : :py:class:`Array`, (`M`,) or scalar float
        The value of the length scale at the specified point.
    """
    if isinstance(X, scipy.ndarray):
        if isinstance(X, scipy.matrix):
            X = scipy.asarray(X, dtype=float)
        return 0.5 * ((l1 + l2) - (l1 - l2) * scipy.tanh((X - x0) / lw))
    else:
        return 0.5 * ((l1 + l2) - (l1 - l2) * mpmath.tanh((X - x0) / lw))


def gauss_warp_arb(X, l1, l2, lw, x0):
    r"""Warps the `X` coordinate with a Gaussian-shaped divot.

    .. math::

        l = l_1 - (l_1 - l_2) \exp\left ( -4\ln 2\frac{(X-x_0)^2}{l_{w}^{2}} \right )

    Parameters
    ----------
    X : :py:class:`Array`, (`M`,) or scalar float
        `M` locations to evaluate length scale at.
    l1 : positive float
        Global value of the length scale.
    l2 : positive float
        Pedestal value of the length scale.
    lw : positive float
        Width of the dip.
    x0 : float
        Location of the center of the dip in length scale.

    Returns
    -------
    l : :py:class:`Array`, (`M`,) or scalar float
        The value of the length scale at the specified point.
    """
    if isinstance(X, scipy.ndarray):
        if isinstance(X, scipy.matrix):
            X = scipy.asarray(X, dtype=float)
        return l1 - (l1 - l2) * scipy.exp(-4.0 * scipy.log(2.0) * (X - x0)**2.0 / (lw**2.0))
    else:
        return l1 - (l1 - l2) * mpmath.exp(-4.0 * mpmath.log(2.0) * (X - x0)**2.0 / (lw**2.0))


def double_tanh_warp(x, n, lcore, lmid, ledge, la, lb, xa, xb):
    r"""Implements a sum-of-tanh warping function and its derivative.

    .. math::

        l = a\tanh\frac{x-x_a}{l_a} + b\tanh\frac{x-x_b}{l_b}

    Parameters
    ----------
    x : float or array of float
        Locations to evaluate the function at.
    n : int
        Derivative order to take. Used for ALL of the points.
    lcore : float
        Core length scale.
    lmid : float
        Intermediate length scale.
    ledge : float
        Edge length scale.
    la : positive float
        Transition of first tanh.
    lb : positive float
        Transition of second tanh.
    xa : float
        Transition of first tanh.
    xb : float
        Transition of second tanh.

    Returns
    -------
    l : float or array
        Warped length scale at the given locations.

    Raises
    ------
    NotImplementedError
        If `n` > 1.
    """
    a, b, c = scipy.dot([[-0.5, 0, 0.5], [0, 0.5, -0.5], [0.5, 0.5, 0]],
                        [[lcore], [ledge], [lmid]])
    a = a[0]
    b = b[0]
    c = c[0]
    if n == 0:
        return a * scipy.tanh((x - xa) / la) + b * scipy.tanh((x - xb) / lb) + c
    elif n == 1:
        return (a**2 / la * (scipy.cosh((x - xa) / la))**(-2.0) +
                b**2 / lb * (scipy.cosh((x - xb) / lb))**(-2.0))
    else:
        raise NotImplementedError("Only derivatives up to order 1 are supported!")


def cubic_bucket_warp(x, n, l1, l2, l3, x0, w1, w2, w3):
    """Warps the length scale with a piecewise cubic "bucket" shape.

    Parameters
    ----------
    x : float or array-like of float
        Locations to evaluate length scale at.
    n : non-negative int
        Derivative order to evaluate. Only first derivatives are supported.
    l1 : positive float
        Length scale to the left of the bucket.
    l2 : positive float
        Length scale in the bucket.
    l3 : positive float
        Length scale to the right of the bucket.
    x0 : float
        Location of the center of the bucket.
    w1 : positive float
        Width of the left side cubic section.
    w2 : positive float
        Width of the bucket.
    w3 : positive float
        Width of the right side cubic section.
    """
    x1 = x0 - w2 / 2.0 - w1 / 2.0
    x2 = x0 + w2 / 2.0 + w3 / 2.0
    x_shift_1 = (x - x1 + w1 / 2.0) / w1
    x_shift_2 = (x - x2 + w3 / 2.0) / w3
    if n == 0:
        return (
            l1 * (x <= (x1 - w1 / 2.0)) + (
                -2.0 * (l2 - l1) * (x_shift_1**3 - 3.0 / 2.0 * x_shift_1**2) + l1
            ) * ((x > (x1 - w1 / 2.0)) & (x < (x1 + w1 / 2.0))) +
            l2 * ((x >= (x1 + w1 / 2.0)) & (x <= x2 - w3 / 2.0)) + (
                -2.0 * (l3 - l2) * (x_shift_2**3 - 3.0 / 2.0 * x_shift_2**2) + l2
            ) * ((x > (x2 - w3 / 2.0)) & (x < (x2 + w3 / 2.0))) +
            l3 * (x >= (x2 + w3 / 2.0))
        )
    elif n == 1:
        return (
            (
                -2.0 * (l2 - l1) * (3 * x_shift_1**2 - 3.0 * x_shift_1) / w1
            ) * ((x > (x1 - w1 / 2.0)) & (x < (x1 + w1 / 2.0))) +
            (
                -2.0 * (l3 - l2) * (3 * x_shift_2**2 - 3.0 * x_shift_2) / w3
            ) * ((x > (x2 - w3 / 2.0)) & (x < (x2 + w3 / 2.0)))
        )
    else:
        raise NotImplementedError("Only up to first derivatives are supported!")


def quintic_bucket_warp(x, n, l1, l2, l3, x0, w1, w2, w3):
    """Warps the length scale with a piecewise quintic "bucket" shape.

    Parameters
    ----------
    x : float or array-like of float
        Locations to evaluate length scale at.
    n : non-negative int
        Derivative order to evaluate. Only first derivatives are supported.
    l1 : positive float
        Length scale to the left of the bucket.
    l2 : positive float
        Length scale in the bucket.
    l3 : positive float
        Length scale to the right of the bucket.
    x0 : float
        Location of the center of the bucket.
    w1 : positive float
        Width of the left side quintic section.
    w2 : positive float
        Width of the bucket.
    w3 : positive float
        Width of the right side quintic section.
    """
    x1 = x0 - w2 / 2.0 - w1 / 2.0
    x2 = x0 + w2 / 2.0 + w3 / 2.0
    x_shift_1 = 2.0 * (x - x1) / w1
    x_shift_3 = 2.0 * (x - x2) / w3
    if n == 0:
        return (
            l1 * (x <= (x1 - w1 / 2.0)) + (
                0.5 * (l2 - l1) * (
                    3.0 / 8.0 * x_shift_1**5 -
                    5.0 / 4.0 * x_shift_1**3 +
                    15.0 / 8.0 * x_shift_1
                ) + (l1 + l2) / 2.0
            ) * ((x > (x1 - w1 / 2.0)) & (x < (x1 + w1 / 2.0))) +
            l2 * ((x >= (x1 + w1 / 2.0)) & (x <= x2 - w3 / 2.0)) + (
                0.5 * (l3 - l2) * (
                    3.0 / 8.0 * x_shift_3**5 -
                    5.0 / 4.0 * x_shift_3**3 +
                    15.0 / 8.0 * x_shift_3
                ) + (l2 + l3) / 2.0
            ) * ((x > (x2 - w3 / 2.0)) & (x < (x2 + w3 / 2.0))) +
            l3 * (x >= (x2 + w3 / 2.0))
        )
    elif n == 1:
        return (
            (
                0.5 * (l2 - l1) * (
                    5.0 * 3.0 / 8.0 * x_shift_1**4 -
                    3.0 * 5.0 / 4.0 * x_shift_1**2 +
                    15.0 / 8.0
                ) / w1
            ) * ((x > (x1 - w1 / 2.0)) & (x < (x1 + w1 / 2.0))) + (
                0.5 * (l3 - l2) * (
                    5.0 * 3.0 / 8.0 * x_shift_3**4 -
                    3.0 * 5.0 / 4.0 * x_shift_3**2 +
                    15.0 / 8.0
                ) / w3
            ) * ((x > (x2 - w3 / 2.0)) & (x < (x2 + w3 / 2.0)))
        )
    else:
        raise NotImplementedError("Only up to first derivatives are supported!")


def betacdf_warp(x, n, alpha, beta):
    """Warp the length scale using the CDF of the Beta distribution

    Parameters
    ----------
    x : float or array-like of float
        Locations to evaluate length scale at. x must be between 0 and 1.
    n : non-negative int or array-like of non-negative int
        Derivative order to evaluate. Only first derivatives are supported.
        The shape of `n` should match the shape of `x`.
    alpha : positive float
        Fist shape parameter for the Beta distribution.
    beta : positive float
        Second shape parameter for the Beta distribution.
    """
    if scipy.any(scipy.logical_or(x > 1, x < 0)):
        raise ValueError('x must be between 0 and 1')
    if scipy.any(scipy.logical_or(alpha <= 0, beta <=0)):
        raise ValueError('alpha and beta must be positive')
    if scipy.any(scipy.logical_or(n < 0, n > 1)):
        raise NotImplementedError('only 0th and 1st derivatives are supported')

    x = scipy.array(x, ndmin=1, dtype=float, copy=False)
    n = scipy.array(n, ndmin=1, dtype=int, copy=False)
    alpha = scipy.array(alpha, ndmin=1, dtype=float, copy=False)
    beta = scipy.array(alpha, ndmin=1, dtype=float, copy=False)

    out = scipy.zeros_like(x)
    mask0 = (n == 0)
    mask1 = (n == 1)

    # 0th-order derivatives. The Beta CDF
    out[mask0] = scipy.special.betainc(alpha, beta, x[mask0])
    
    # 1st-order derivatives. The Beta PDF
    # See scipy.stats._continuous_distns.py#L380
    x1 = x[mask1]
    out1 = scipy.special.xlog1py(beta-1.0, -x1)
    out1 += scipy.special.xlogy(alpha-1.0, x1)
    out1 -= scipy.special.betaln(alpha, beta)
    out[mask1] = scipy.exp(out1)

    return out

    
