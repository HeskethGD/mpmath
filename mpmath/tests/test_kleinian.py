import importlib

import pytest

from mpmath import (
    diff, kleinian_p, kleinian_sigma, kleinian_zeta, log, mp,
    weierp, weierpprime, weiersigma, weierzeta,
)


def test_kleinian_genus_one_matches_weierstrass_functions():
    mp.dps = 35
    omega1 = mp.mpf('0.7')
    omega2 = mp.mpc('0.2', '0.9')
    tau = [[omega2 / omega1]]
    omega = [[2 * omega1]]
    unused_q, unused_scale, kappa = mp._weierzeta_data(omega1, omega2)
    kappa = [[kappa]]
    half = mp.mpf('0.5')
    characteristic = ([half], [half])
    points = (mp.mpc('0.23', '0.07'), mp.mpc('-0.31', '0.04'))

    sigma_ratios = []
    for u in points:
        args = ([u], omega, tau, kappa)
        sigma_ratios.append(
            kleinian_sigma(*args, characteristic)
            / weiersigma(u, omega1=omega1, omega2=omega2))
        assert mp.almosteq(
            kleinian_zeta(*args, characteristic)[0],
            weierzeta(u, omega1=omega1, omega2=omega2))
        assert mp.almosteq(
            kleinian_p(*args, (0, 0), characteristic),
            weierp(u, omega1=omega1, omega2=omega2))
        assert mp.almosteq(
            kleinian_p(*args, (0, 0, 0), characteristic),
            weierpprime(u, omega1=omega1, omega2=omega2))
    assert mp.almosteq(sigma_ratios[0], sigma_ratios[1])


def test_kleinian_derivative_identities_and_batched_p():
    mp.dps = 25
    omega = [[mp.mpf('1.2'), mp.mpf('0.1')],
             [mp.mpf('0.05'), mp.mpf('0.9')]]
    tau = [[mp.mpc('0.1', '1.0'), mp.mpc('-0.08', '0.05')],
           [mp.mpc('-0.08', '0.05'), mp.mpc('-0.15', '1.2')]]
    kappa = [[mp.mpc('0.3', '0.1'), mp.mpc('-0.12', '0.04')],
             [mp.mpc('-0.12', '0.04'), mp.mpc('0.2', '-0.03')]]
    characteristic = ([mp.mpf('0.5'), 0], [0, mp.mpf('0.5')])
    u = [mp.mpc('0.17', '0.03'), mp.mpc('-0.11', '0.02')]

    def sigma_at(values):
        return kleinian_sigma(
            values, omega, tau, kappa, characteristic, constant=3)

    zeta = kleinian_zeta(u, omega, tau, kappa, characteristic)
    for i in range(2):
        expected = diff(
            lambda value: log(sigma_at(
                [value if j == i else u[j] for j in range(2)])),
            u[i])
        assert mp.almosteq(zeta[i], expected)

    requested = ((0, 0), (0, 1), (1, 0), (0, 0, 1))
    values = kleinian_p(
        u, omega, tau, kappa, requested, characteristic)
    assert len(values) == len(requested)
    for indices, value in zip(requested, values):
        assert mp.almosteq(
            value, kleinian_p(
                u, omega, tau, kappa, indices, characteristic))
    assert mp.almosteq(values[1], values[2])

    p01_from_zeta = -diff(
        lambda value: kleinian_zeta(
            [u[0], value], omega, tau, kappa, characteristic)[0],
        u[1])
    assert mp.almosteq(values[1], p01_from_zeta)
    p001_from_p = diff(
        lambda value: kleinian_p(
            [u[0], value], omega, tau, kappa, (0, 0), characteristic),
        u[1])
    assert mp.almosteq(values[3], p001_from_p)

    doubled = kleinian_sigma(
        u, omega, tau, kappa, characteristic, constant=2)
    ordinary = kleinian_sigma(
        u, omega, tau, kappa, characteristic)
    assert mp.almosteq(doubled, 2 * ordinary)


def test_kleinian_p_periodicity_and_sigma_parity():
    # Eilbeck, Enolskii & Leykin (2000), Definition 3.7: Kleinian P-functions
    # are invariant under the full period lattice. Their periods are written
    # as 2*omega and 2*omega_prime; this test uses our full-period convention.
    mp.dps = 35
    omega = mp.matrix([[mp.mpf('1.2'), mp.mpf('0.1')],
                       [mp.mpf('0.05'), mp.mpf('0.9')]])
    tau = mp.matrix([
        [mp.mpc('0.1', '1.0'), mp.mpc('-0.08', '0.05')],
        [mp.mpc('-0.08', '0.05'), mp.mpc('-0.15', '1.2')],
    ])
    kappa = mp.matrix([
        [mp.mpc('0.3', '0.1'), mp.mpc('-0.12', '0.04')],
        [mp.mpc('-0.12', '0.04'), mp.mpc('0.2', '-0.03')],
    ])
    u = mp.matrix([mp.mpc('0.17', '0.03'), mp.mpc('-0.11', '0.02')])
    half = mp.mpf('0.5')
    odd_characteristic = ([half, 0], [half, 0])
    omega_prime = omega * tau
    shifted_u = (
        u + omega * mp.matrix([1, -1])
        + omega_prime * mp.matrix([1, 1])
    )
    indices = ((0, 0), (0, 1), (1, 1),
               (0, 0, 0), (0, 1, 1), (1, 1, 1))

    values = kleinian_p(
        u, omega, tau, kappa, indices, odd_characteristic)
    shifted_values = kleinian_p(
        shifted_u, omega, tau, kappa, indices, odd_characteristic)
    for value, shifted_value in zip(values, shifted_values):
        assert abs(value - shifted_value) < 100 * mp.eps * max(1, abs(value))

    # The genus-two sigma characteristic used by EEL is odd.
    assert mp.almosteq(
        kleinian_sigma(-u, omega, tau, kappa, odd_characteristic),
        -kleinian_sigma(u, omega, tau, kappa, odd_characteristic))


def test_kleinian_bernatska_genus_four_example():
    # J. Bernatska, "Computation of P-Functions on Plane Algebraic Curves",
    # Journal of Experimental Mathematics 2(1) (2026), 114--154,
    # doi:10.56994/JXM.002.001.005, Example 1a. Period data are printed to
    # only 6--7 digits, so this checks conventions with a loose tolerance.
    mp.dps = 20
    c = mp.mpc
    omega = [
        [c('-1.303573', '0.207439'), c('0.848115', '-0.306788'),
         c('0.0166625', '0.063503'), c('-0.035439', '-0.017840')],
        [c('0.073367', '-0.003075'), c('-0.083799', '0.019801'),
         c('0.005372', '-0.014707'), c('-0.007363', '-0.005200')],
        [c('-0.003985', '-0.000333'), c('0.008037', '-0.001042'),
         c('-0.003023', '0.002108'), c('-0.001651', '-0.000958')],
        [c('0.000208', '0.000046'), c('-0.000751', '0.000028'),
         c('0.000856', '0.000006'), c('-0.000350', '-0.000098')],
    ]
    tau = [
        [c('0.416960', '1.348235'), c('-0.019631', '0.866637'),
         c('0.043442', '0.592788'), c('0.013536', '0.360353')],
        [c('-0.019631', '0.866637'), c('-0.401986', '1.468494'),
         c('0.090347', '0.771653'), c('0.020075', '0.430424')],
        [c('0.043442', '0.592788'), c('0.090347', '0.771653'),
         c('0.276110', '1.677311'), c('-0.019449', '0.549477')],
        [c('0.013536', '0.360353'), c('0.020075', '0.430424'),
         c('-0.019449', '0.549477'), c('-0.241045', '0.959518')],
    ]
    kappa = [
        [c('-26.150273', '5.226639'), c('-113.639362', '91.745099'),
         c('815.048336', '59.142845'), c('2796.548807', '-2715.208601')],
        [c('-113.639362', '91.745099'), c('2527.918193', '333.2000001'),
         c('6691.213749', '-15142.962600'),
         c('-19805.451622', '-22245.716646')],
        [c('815.048336', '59.142845'), c('6691.213749', '-15142.962600'),
         c('-501204.576087', '-151451.871496'),
         c('-1572965.591976', '1699015.043174')],
        [c('2796.548807', '-2715.208601'),
         c('-19805.451622', '-22245.716646'),
         c('-1572965.591976', '1699015.043174'),
         c('-403196.119224', '19865411.502694')],
    ]
    u = [c('-1.181275', '0.204397'), c('0.0736446', '-0.0379656'),
         c('-0.00480803', '0.00259696'),
         c('0.000610581', '-0.0000580815')]
    characteristic = ([mp.mpf('0.5')] * 4,
                      [0, mp.mpf('0.5'), 0, mp.mpf('0.5')])
    indices = ((0, 0), (0, 1), (0, 0, 0), (0, 0, 1))
    expected = (c(-5, 4), c(44, 46),
                c('105.080464', '-182.470785'),
                c('-106.944505', '-1805.409841'))
    values = kleinian_p(
        u, omega, tau, kappa, indices, characteristic)
    for value, reference in zip(values, expected):
        assert abs(value - reference) < 8


def test_kleinian_validation():
    mp.dps = 20
    tau = [[1j, 0], [0, 1.2j]]
    omega = [[1, 0], [0, 1]]
    kappa = [[1, 0], [0, 1]]
    u = [mp.mpf('0.1'), mp.mpf('0.2')]

    with pytest.raises(ValueError, match="omega must be invertible"):
        kleinian_sigma(u, [[1, 0], [0, 0]], tau, kappa)
    with pytest.raises(ValueError, match="omega must be a square matrix"):
        kleinian_sigma(u, object(), tau, kappa)
    with pytest.raises(ValueError, match="nonempty square matrix"):
        kleinian_sigma(u, [[1, 0]], tau, kappa)
    with pytest.raises(ValueError, match="entries must be finite"):
        kleinian_sigma(u, [[1, 0], [0, mp.inf]], tau, kappa)
    with pytest.raises(ValueError, match="tau must be a 2 by 2 matrix"):
        kleinian_sigma(u, omega, [[1j]], kappa)
    with pytest.raises(ValueError, match="kappa must be a 2 by 2 matrix"):
        kleinian_sigma(u, omega, tau, [[1]])
    with pytest.raises(ValueError, match="kappa must be symmetric"):
        kleinian_sigma(u, omega, tau, [[1, 1], [0, 1]])
    with pytest.raises(ValueError, match="indices must contain"):
        kleinian_p(u, omega, tau, kappa, 1)
    with pytest.raises(ValueError, match="at least one"):
        kleinian_p(u, omega, tau, kappa, [])
    with pytest.raises(ValueError, match="length 2 or 3"):
        kleinian_p(u, omega, tau, kappa, (0,))
    with pytest.raises(ValueError, match="between 0 and 1"):
        kleinian_p(u, omega, tau, kappa, (0, 2))
    with pytest.raises(ValueError, match="between 0 and 1"):
        kleinian_p(u, omega, tau, kappa, [(0, mp.mpf(1))])
    with pytest.raises(ValueError, match="constant must be finite"):
        kleinian_sigma(u, omega, tau, kappa, constant=mp.inf)


def test_kleinian_theta_divisor(monkeypatch):
    module = importlib.import_module("mpmath.functions.kleinian")

    monkeypatch.setattr(
        module, "_rtheta_derivatives",
        lambda ctx, v, tau, characteristic, derivatives:
            (ctx.zero,) * len(derivatives))
    with pytest.raises(ZeroDivisionError, match="theta divisor"):
        module._theta_log_jet(mp, (0,), ((1j,),), None, 1)
