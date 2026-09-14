Abelian functions
-----------------

Riemann theta functions generalize the Jacobi theta functions from one
complex variable to several. They arose in Riemann's nineteenth-century
theory of Abelian functions and compact Riemann surfaces. Today they are
fundamental tools in complex analysis and algebraic geometry, and occur in
finite-gap and quasiperiodic solutions of integrable systems.

Theta functions are quasiperiodic. For integer vectors
:math:`m,n\in\mathbb Z^g`, the zero-characteristic function satisfies

.. math ::

    \theta(z+m+\tau n\mid\tau)
    = \exp\!\left(-\pi i n^T\tau n-2\pi i n^Tz\right)
      \theta(z\mid\tau).

Although theta functions are quasiperiodic, suitable ratios and combinations
of them give multiply-periodic Abelian functions. In genus two, such Abelian
functions are meromorphic functions of two complex variables with a period
lattice of rank four, generated in normalized coordinates by the columns of
:math:`I_2` and :math:`\tau`.

Kleinian sigma, zeta and P-functions extend the corresponding Weierstrass
functions to higher genus. The sigma function combines Riemann theta with
unnormalized period data and an exponential factor; its logarithmic
derivatives give the zeta and P-functions described below.


Riemann theta functions
.......................

.. autofunction:: mpmath.rtheta

The following plots show two real slices and the modulus over two real
variables for genus-two period matrices. Similar slices and surfaces are
illustrated in `DLMF section 21.4 <https://dlmf.nist.gov/21.4>`_.

.. plot::

   import matplotlib.pyplot as plt
   from mpmath import j, plot, re, rtheta

   tau = [[j, -0.5], [-0.5, j]]
   curves = [
       lambda x: re(rtheta([x, x/2], tau)),
       lambda x: re(rtheta([x, 2*x], tau)),
   ]
   fig, ax = plt.subplots()
   plot(curves, [-2, 2], axes=ax)
   ax.legend([r"$z=(x,x/2)$", r"$z=(x,2x)$"])

.. plot::

   import matplotlib.pyplot as plt
   from mpmath import j, rtheta, splot

   tau = [[j, 0.5], [0.5, j]]
   fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
   surface = lambda x, y: abs(rtheta([x, y], tau))
   splot(surface, [-1, 1], [-1, 1], points=35, keep_aspect=False,
         axes=ax, plot3d_kwargs={"cmap": "viridis"})
   ax.set_zlabel(r"$|\theta(z\mid\tau)|$")


Riemann theta jets
..................

.. autofunction:: mpmath.rtheta_jet


Hyperelliptic period data
.........................

For an odd-degree hyperelliptic curve with distinct real branch points,
:func:`~mpmath.hyperelliptic_periods` constructs full first-kind periods and
the normalized Riemann matrix directly from the polynomial coefficients. It
can also construct the associated canonical second-kind periods and
:math:`\varkappa=\eta\omega^{-1}`. The real cycle arrangement follows Baker
and Bernatska, while the arbitrary-genus algebraic second-kind basis is
equation (1.3) of [BEL1997]_.

.. autofunction:: mpmath.hyperelliptic_periods


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

The supplied period data must be mutually consistent. For the currently
supported real odd-degree hyperelliptic curves,
:func:`~mpmath.hyperelliptic_periods` can construct ``omega``, ``tau`` and
``kappa`` together by setting ``second_kind=True``. The Kleinian functions do
not yet infer the Riemann characteristic or canonical sigma normalization
constant from the curve.

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
