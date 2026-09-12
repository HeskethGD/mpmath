import pytest

from mpmath import diff, exp, j, jtheta, mp, pi, rtheta


def test_rtheta_genus_one_jtheta_characteristics():
    mp.dps = 30
    w = mp.mpc('0.3', '0.1')
    tau = mp.mpc('0.2', '0.9')
    q = exp(pi * j * tau)
    z = w / pi
    cases = [
        (([mp.mpf('0.5')], [mp.mpf('0.5')]), -jtheta(1, w, q)),
        (([mp.mpf('0.5')], [0]), jtheta(2, w, q)),
        (([0], [0]), jtheta(3, w, q)),
        (([0], [mp.mpf('0.5')]), jtheta(4, w, q)),
    ]
    for characteristic, expected in cases:
        assert mp.almosteq(rtheta([z], [[tau]], characteristic), expected)


def test_rtheta_diagonal_factorisation():
    mp.dps = 30
    z = [mp.mpc('0.1', '0.03'), mp.mpc('-0.2', '0.04')]
    tau = [[mp.mpc('0.1', '0.8'), 0], [0, mp.mpc('-0.2', '1.1')]]
    value = rtheta(z, tau)
    expected = (rtheta([z[0]], [[tau[0][0]]])
                * rtheta([z[1]], [[tau[1][1]]]))
    assert mp.almosteq(value, expected)


def test_rtheta_parity_and_characteristic_zero():
    mp.dps = 30
    tau = [[1j, mp.mpc('0.1', '0.05')],
           [mp.mpc('0.1', '0.05'), mp.mpc('0.2', '1.2')]]
    z = [mp.mpc('0.13', '0.02'), mp.mpc('-0.07', '0.01')]
    assert mp.almosteq(rtheta(z, tau), rtheta([-z[0], -z[1]], tau))
    odd = ([mp.mpf('0.5'), 0], [mp.mpf('0.5'), 0])
    assert abs(rtheta([0, 0], tau, odd)) < mp.eps * 10


def test_rtheta_quasiperiodicity():
    mp.dps = 30
    tau = [[mp.mpc('0.1', '0.9'), mp.mpc('0.05', '0.02')],
           [mp.mpc('0.05', '0.02'), mp.mpc('-0.1', '1.1')]]
    z = [mp.mpc('0.12', '0.02'), mp.mpc('-0.08', '0.03')]
    value = rtheta(z, tau)
    assert mp.almosteq(rtheta([z[0] + 1, z[1] - 2], tau), value)

    k = [1, -1]
    shifted = [z[i] + sum(tau[i][j_] * k[j_] for j_ in range(2))
               for i in range(2)]
    quadratic = sum(k[i] * tau[i][j_] * k[j_]
                    for i in range(2) for j_ in range(2))
    linear = sum(k[i] * z[i] for i in range(2))
    multiplier = exp(-pi * j * quadratic - 2 * pi * j * linear)
    assert mp.almosteq(rtheta(shifted, tau), multiplier * value)


def test_rtheta_derivatives():
    mp.dps = 25
    tau = [[mp.mpc('0.1', '0.9'), mp.mpc('0.05', '0.02')],
           [mp.mpc('0.05', '0.02'), mp.mpc('-0.1', '1.1')]]
    z = [mp.mpc('0.12', '0.02'), mp.mpc('-0.08', '0.03')]
    dz0 = rtheta(z, tau, derivative=(1, 0))
    expected = diff(lambda value: rtheta([value, z[1]], tau), z[0])
    assert mp.almosteq(dz0, expected)

    w = mp.mpc('0.2', '0.04')
    tau1 = mp.mpc('0.15', '0.85')
    q = exp(pi * j * tau1)
    for order in (1, 2):
        assert mp.almosteq(
            rtheta([w / pi], [[tau1]], derivative=order),
            pi ** order * jtheta(3, w, q, derivative=order))


def test_rtheta_precision_doubling_genus_three():
    mp.dps = 100
    tau = [[1.1j, 0.04j, 0.02j],
           [0.04j, 1.2j, 0.03j],
           [0.02j, 0.03j, 1.3j]]
    z = [mp.mpc('0.1', '0.02'), mp.mpc('-0.15', '0.01'),
         mp.mpc('0.07', '-0.03')]
    value = rtheta(z, tau, derivative=(1, 0, 1))
    with mp.workdps(130):
        reference = rtheta(z, tau, derivative=(1, 0, 1))
    assert mp.almosteq(value, reference, rel_eps=mp.mpf('1e-98'))


def test_rtheta_validation():
    mp.dps = 20
    with pytest.raises(ValueError, match="vector"):
        rtheta(None, [[1j]])
    with pytest.raises(ValueError, match="vector"):
        rtheta(mp.matrix([[0, 0]]), [[1j]])
    with pytest.raises(ValueError, match="finite"):
        rtheta([mp.inf], [[1j]])
    with pytest.raises(ValueError, match="square"):
        rtheta([0], None)
    with pytest.raises(ValueError, match="square"):
        rtheta([0], [[1j, 0]])
    with pytest.raises(ValueError, match="finite"):
        rtheta([0], [[mp.inf + 1j]])
    with pytest.raises(ValueError, match="length"):
        rtheta([0], [[1j, 0], [0, 1j]])
    with pytest.raises(ValueError, match="symmetric"):
        rtheta([0, 0], [[1j, 1], [0, 1j]])
    with pytest.raises(ValueError, match="positive definite"):
        rtheta([0, 0], [[1j, 0], [0, -1j]])
    with pytest.raises(ValueError, match="real"):
        rtheta([0], [[1j]], characteristic=([1j], [0]))
    with pytest.raises(ValueError, match="pair"):
        rtheta([0], [[1j]], characteristic=([0],))
    with pytest.raises(ValueError, match="length"):
        rtheta([0, 0], [[1j, 0], [0, 1j]], derivative=(1,))
    with pytest.raises(ValueError, match="genus 1"):
        rtheta([0, 0], [[1j, 0], [0, 1j]], derivative=1)
    with pytest.raises(ValueError, match="multi-index"):
        rtheta([0], [[1j]], derivative=None)
    with pytest.raises(ValueError, match="nonnegative"):
        rtheta([0], [[1j]], derivative=(1.0,))
    with pytest.raises(ValueError, match="nonnegative"):
        rtheta([0], [[1j]], derivative=(-1,))


def test_rtheta_nearly_symmetric_input():
    mp.dps = 20
    delta = mp.eps / 4
    tau = [[1j, mp.mpc('0.1', '0.02')],
           [mp.mpc('0.1', '0.02') + delta, 1.2j]]
    symmetric = [[1j, mp.mpc('0.1', '0.02') + delta / 2],
                 [mp.mpc('0.1', '0.02') + delta / 2, 1.2j]]
    assert mp.almosteq(rtheta([0.1, 0.2], tau),
                       rtheta([0.1, 0.2], symmetric))
