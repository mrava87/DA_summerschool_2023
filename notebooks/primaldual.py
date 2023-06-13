import numpy as np
import pyproximal
from pyproximal.ProxOperator import _check_tau
from pyproximal import ProxOperator
from pyproximal.proximal import L2
from pyproximal.proximal import *
from pyproximal.optimization.primaldual import PrimalDual


def callback(x, xtrue, xhist, errhist):
    """Callback for PnP
    """
    xhist.append(x)
    errhist.append(np.linalg.norm(x - xtrue) / np.linalg.norm(xtrue))


class _Denoise(ProxOperator):
    r"""Denoiser of choice

    Parameters
    ----------
    denoiser : :obj:`func`
        Denoiser (must be a function with two inputs, the first is the signal
        to be denoised, the second is the `tau` constant of the y-update in
        the proximal algorithm of choice)
    dims : :obj:`tuple`
        Dimensions used to reshape the vector ``x`` in the ``prox`` method
        prior to calling the ``denoiser``

    """

    def __init__(self, denoiser, dims):
        super().__init__(None, False)
        self.denoiser = denoiser
        self.dims = dims

    def __call__(self, x):
        return 0.

    @_check_tau
    def prox(self, x, tau):
        x = x.reshape(self.dims)
        xden = self.denoiser(x, tau)
        return xden.ravel()


def PlugAndPlay_PrimalDual(proxf, denoiser, A, dims, x0, tau, mu, niter=10,
                           gfirst=True, callback=None, show=False):
    r"""Plug-and-Play Priors with Primal-Dual optimization

    Solves the following minimization problem using the Primal-Dual algorithm:

    .. math::

        \mathbf{x},\mathbf{z}  = \argmin_{\mathbf{x}}
        f(\mathbf{x}) + \lambda g(\mathbf{x})

    where :math:`f(\mathbf{x})` is a function that has a known proximal
    operator where :math:`g(\mathbf{x})` is a function acting as implicit
    prior. Implicit means that no explicit function should be defined: instead,
    a denoising algorithm of choice is used. See Notes for details.

    Parameters
    ----------
    proxf : :obj:`pyproximal.ProxOperator`
        Proximal operator of f function
    denoiser : :obj:`func`
        Denoiser (must be a function with two inputs, the first is the signal
        to be denoised, the second is the tau constant of the y-update in
        PlugAndPlay)
    A : :obj:`pylops.LinearOperator`
        Linear operator of the denoiser, usually the Identity operator
    dims : :obj:`tuple`
        Dimensions used to reshape the vector ``x`` in the ``prox`` method
        prior to calling the ``denoiser``
    x0 : :obj:`numpy.ndarray`
        Initial vector
    tau : :obj:`float` or :obj:`np.ndarray`
        Stepsize of subgradient of :math:`f`. This can be constant
        or function of iterations (in the latter cases provided as np.ndarray)
    mu : :obj:`float` or :obj:`np.ndarray`
        Stepsize of subgradient of :math:`g^*`. This can be constant
        or function of iterations (in the latter cases provided as np.ndarray)
    niter : :obj:`int`, optional
        Number of iterations of iterative scheme
    gfirst : :obj:`bool`, optional
        Apply Proximal of operator ``g`` first (``True``) or Proximal of
        operator ``f`` first (``False``)
    callback : :obj:`callable`, optional
        Function with signature (``callback(x)``) to call after each iteration
        where ``x`` is the current model vector
    show : :obj:`bool`, optional
        Display iterations log

    Returns
    -------
    x : :obj:`numpy.ndarray`
        Inverted model

    Notes
    -----
    For more details see :func:``pyproximal.optimization.pnp.PlugAndPlay``.

    """
    # Denoiser
    proxpnp = _Denoise(denoiser, dims=dims)

    return PrimalDual(proxf, proxpnp, A, x0=x0, tau=tau, mu=mu,
                      niter=niter, callback=callback, gfirst=gfirst,
                      show=show)