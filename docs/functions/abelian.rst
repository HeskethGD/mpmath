Abelian functions
-----------------

Riemann theta functions are fundamental building blocks for Abelian
functions associated with complex tori and compact Riemann surfaces. The
Kleinian functions below combine a Riemann theta function with unnormalized
period data.


Riemann theta functions
.......................

.. autofunction:: mpmath.rtheta


Kleinian functions
...................

The Kleinian functions use unnormalized Abelian coordinates ``u``. Their
period data are the first-kind a-period matrix ``omega``, the normalized
Riemann matrix ``tau``, and the symmetric matrix ``kappa`` equal to
:math:`\eta\omega^{-1}`. These matrices must all have size :math:`g\times g`.
The convention is

.. math::

    \tau = \omega^{-1}\omega', \qquad v = \omega^{-1}u.

Thus, ``omega`` is the full a-period matrix in this interface; it is twice a
half-period matrix when the latter convention is used. The characteristic is
specified as ``(a, b)`` using the same literal convention as
:func:`~mpmath.rtheta`.

The supplied period data must be mutually consistent. These functions do not
yet construct period matrices or the canonical normalization constant of the
sigma function from a curve.

The sigma and P-function definitions follow [Bernatska2026]_. Signs for
second-kind differentials are not universal. In the half-period notation of
[EEL2000]_, the first- and second-kind period matrices are written
:math:`2\omega` and :math:`2\eta`, and the sigma exponential has the opposite
sign. Data in that convention are converted to this interface by using
``omega = 2*omega_EEL`` and
:math:`\varkappa=-2\eta_{\rm EEL}(2\omega_{\rm EEL})^{-1}`.

For the classical development of Abelian, theta, sigma, and multiply periodic
functions, see [BakerAbel]_ and [BakerMultiply]_.

.. autofunction:: mpmath.kleinian_sigma

.. autofunction:: mpmath.kleinian_zeta

.. autofunction:: mpmath.kleinian_p
