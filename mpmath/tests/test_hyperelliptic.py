import pytest

from mpmath import (
    hyperelliptic_kleinian_data, hyperelliptic_periods, kleinian_p,
    kleinian_sigma, kleinian_zeta, mp, weierp, weiersigma, weierzeta,
)
from mpmath.functions import hyperelliptic


def test_hyperelliptic_periods_genus_one_lemniscatic_curve():
    # y^2 = x^3-x has the square period lattice, hence tau = i in the
    # automatically selected real-branch basis.
    mp.dps = 30
    omega, omega_prime, tau = hyperelliptic_periods([0, -1, 0, 1])
    assert omega.rows == omega.cols == 1
    assert mp.almosteq(omega_prime[0, 0], (omega * tau)[0, 0])
    assert mp.almosteq(tau[0, 0], 1j)


def test_hyperelliptic_periods_genus_two_relations():
    # P(x) = x(x^2-1)(x^2-4) has five distinct real branch points.
    mp.dps = 30
    omega, omega_prime, tau = hyperelliptic_periods(
        [0, 4, 0, -5, 0, 1], method="real")
    assert omega.rows == omega.cols == 2
    assert mp.norm(omega * tau - omega_prime) < mp.mpf('1e-28')
    assert mp.norm(tau - tau.T) < mp.mpf('1e-28')
    assert mp.im(tau[0, 0]) > 0
    assert mp.im(tau[1, 1]) > 0
    assert (mp.im(tau[0, 0]) * mp.im(tau[1, 1])
            - mp.im(tau[0, 1]) ** 2) > 0


def test_hyperelliptic_periods_genus_three():
    # P(x) = x(x^2-1)(x^2-4)(x^2-9). This primarily checks that the cycle
    # construction and differential basis extend beyond genus 2.
    mp.dps = 25
    omega, omega_prime, tau = hyperelliptic_periods(
        [0, -36, 0, 49, 0, -14, 0, 1])
    assert omega.rows == omega.cols == 3
    assert mp.norm(omega * tau - omega_prime) < mp.mpf('1e-23')
    assert mp.norm(tau - tau.T) < mp.mpf('1e-23')
    imaginary_tau = mp.matrix([
        [mp.im(tau[i, j]) for j in range(3)] for i in range(3)])
    mp.cholesky(imaginary_tau)


def test_hyperelliptic_second_kind_genus_one():
    # BEL (1997), equation (1.3), gives dr = x dx/(4y) for this odd cubic.
    # Its period ratio is the exponential coefficient in the conventional
    # Weierstrass sigma and zeta functions.
    mp.dps = 30
    data = hyperelliptic_periods([0, -1, 0, 1], second_kind=True)
    omega, omega_prime, eta, eta_prime, tau, kappa = data
    omega1 = omega[0, 0] / 2
    omega2 = omega_prime[0, 0] / 2
    unused_q, unused_scale, expected_kappa = mp._weierzeta_data(
        omega1, omega2)
    assert mp.almosteq(tau[0, 0], omega2 / omega1)
    assert mp.almosteq(kappa[0, 0], expected_kappa)
    assert mp.almosteq(eta[0, 0], kappa[0, 0] * omega[0, 0])
    assert mp.almosteq(
        omega[0, 0] * eta_prime[0, 0]
        - omega_prime[0, 0] * eta[0, 0],
        2 * mp.pi * 1j)


def test_hyperelliptic_second_kind_genus_two_legendre_relation():
    # Buchstaber, Enolskii & Leykin (1997), Lemma 1.1. The paper uses
    # half-period notation and the opposite sign for eta; this is the same
    # identity in our full-period, Bernatska-sign convention.
    mp.dps = 30
    data = hyperelliptic_periods(
        [-15, 20.5, 26.5, -13.5, -3.5, 1], second_kind=True)
    omega, omega_prime, eta, eta_prime, unused_tau, kappa = data
    assert mp.norm(kappa - kappa.T) < mp.mpf('1e-28')
    periods = mp.matrix(4)
    periods[:2, :2] = omega
    periods[:2, 2:] = omega_prime
    periods[2:, :2] = eta
    periods[2:, 2:] = eta_prime
    symplectic = mp.matrix([
        [0, 0, -1, 0],
        [0, 0, 0, -1],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ])
    expected = 2 * mp.pi * 1j * symplectic
    assert mp.norm(
        periods * symplectic * periods.T - expected) < mp.mpf('1e-27')


def test_hyperelliptic_kleinian_data_genus_one():
    # In genus one, Bernatska's sum of branch-point characteristics gives the
    # classical odd characteristic [1/2, 1/2]. The resulting logarithmic
    # derivatives and the Schur-normalized sigma agree with the conventional
    # Weierstrass functions.
    mp.dps = 30
    omega, tau, kappa, characteristic = hyperelliptic_kleinian_data(
        [0, -1, 0, 1])
    assert characteristic == ((0.5,), (0.5,))
    omega1 = omega[0, 0] / 2
    omega2 = (omega * tau)[0, 0] / 2
    points = (mp.mpf('0.2'), mp.mpc('0.3', '0.04'))
    for point in points:
        arguments = ([point], omega, tau, kappa)
        assert mp.almosteq(
            kleinian_p(*arguments, (0, 0), characteristic),
            weierp(point, omega1=omega1, omega2=omega2))
        assert mp.almosteq(
            kleinian_zeta(*arguments, characteristic)[0],
            weierzeta(point, omega1=omega1, omega2=omega2))
        assert mp.almosteq(
            kleinian_sigma(
                *arguments, characteristic,
                normalization="hyperelliptic"),
            weiersigma(point, omega1=omega1, omega2=omega2))


@pytest.mark.parametrize("coefficients", [
    [0, 16, 0, -20, 0, 4],
    [0, -36, 0, 49, 0, -14, 0, 1],
])
def test_hyperelliptic_sigma_schur_normalization(coefficients):
    # Buchstaber, Enolskii & Leykin (1997), Definition 1: the fundamental
    # sigma function starts with delta(u)=det(u[i+j-1]). The coefficient of
    # u_m**m is the sign of the reversing permutation, m=floor((g+1)/2).
    mp.dps = 35
    omega, tau, kappa, characteristic = hyperelliptic_kleinian_data(
        coefficients)
    genus = omega.rows
    degree = (genus + 1) // 2
    coordinate = degree - 1

    def sigma_on_axis(value):
        u = [mp.zero] * genus
        u[coordinate] = value
        return kleinian_sigma(
            u, omega, tau, kappa, characteristic,
            normalization="hyperelliptic")

    derivative = mp.diff(sigma_on_axis, 0, degree)
    sign = -1 if (degree * (degree - 1) // 2) & 1 else 1
    assert mp.almosteq(derivative, sign * mp.factorial(degree))


def test_hyperelliptic_kleinian_data_genus_two_periodicity():
    # Eilbeck, Enolskii & Leykin (2000), Definition 3.7, gives full-lattice
    # periodicity. BEL (1997), equation (3.9), gives the genus-two cubic
    # identity checked below. Together they test that periods, kappa and the
    # Riemann characteristic constructed from one curve work coherently.
    mp.dps = 40
    coefficients = [0, 16, 0, -20, 0, 4]
    omega, tau, kappa, characteristic = hyperelliptic_kleinian_data(
        coefficients)
    half = 0.5
    assert characteristic == ((half, half), (0, half))
    u = mp.matrix([mp.mpc('0.13', '0.02'), mp.mpc('-0.08', '0.01')])
    shifted = (
        u + omega * mp.matrix([1, -1])
        + omega * tau * mp.matrix([1, 1])
    )
    indices = ((0, 0), (0, 1), (1, 1), (0, 0, 1), (1, 1, 1))
    values = kleinian_p(
        u, omega, tau, kappa, indices, characteristic)
    shifted_values = kleinian_p(
        shifted, omega, tau, kappa, indices, characteristic)
    for value, shifted_value in zip(values, shifted_values):
        assert abs(value - shifted_value) < (
            100 * mp.eps * max(1, abs(value)))
    assert mp.almosteq(
        kleinian_sigma(-u, omega, tau, kappa, characteristic),
        -kleinian_sigma(u, omega, tau, kappa, characteristic))
    p11, p12, p22, unused_p112, p222 = values
    cubic = (
        4 * p22 ** 3 + 4 * p22 * p12 + 4 * p11
        + coefficients[4] * p22 ** 2 + coefficients[3] * p22
        + coefficients[2]
    )
    assert abs(p222 ** 2 - cubic) < (
        100 * mp.eps * max(1, abs(cubic)))


@pytest.mark.parametrize("coefficients", [
    [24, 14, -13, -2, 1],
    [1, 0, 0, 0, 1],
])
def test_hyperelliptic_even_degree_genus_one_cubic(coefficients):
    # BEL (1997), the genus-one discussion following equation (3.15), gives
    # this cubic for a general quartic model. It checks the even-degree
    # periods, second-kind data, Riemann characteristic and derivatives
    # together for real and nonreal branch points without first transforming
    # the curve to an odd-degree model.
    mp.dps = 40
    omega, tau, kappa, characteristic = hyperelliptic_kleinian_data(
        coefficients)
    u = [mp.mpc('0.2', '0.03')]
    p11, p111 = kleinian_p(
        u, omega, tau, kappa, ((0, 0), (0, 0, 0)), characteristic)
    lambda0, lambda1, lambda2, lambda3, lambda4 = coefficients
    cubic = (
        4 * p11 ** 3 + lambda2 * p11 ** 2
        + (lambda1 * lambda3 - 4 * lambda4 * lambda0) * p11 / 4
        + (lambda0 * lambda3 ** 2
           + lambda4 * (lambda1 ** 2 - 4 * lambda2 * lambda0)) / 16
    )
    assert abs(p111 ** 2 - cubic) < (
        100 * mp.eps * max(1, abs(cubic)))


def test_hyperelliptic_even_degree_pari_oracle():
    # PARI/GP 2.18.1 alpha, hyperellperiods(P, 1), at 70 decimal digits for
    # P=(x+3)(x+2)(x+1)(x-1)(x-2)(x-4). As in the odd-degree oracle, PARI's
    # interleaved cycles have the opposite intersection orientation. The
    # anti-symplectic matrix below relates its independent lattice to ours.
    mp.dps = 30
    omega, omega_prime, unused_tau = hyperelliptic_periods(
        [-48, -4, 64, 5, -17, -1, 1])
    periods = mp.matrix(2, 4)
    periods[:, :2] = omega
    periods[:, 2:] = omega_prime
    r = mp.mpf
    pari_periods = mp.matrix([
        [r('0.464901227905802907214915155765884188') * 1j,
         -r('0.531986563561318428544478628778286555') * 1j,
         r('0.645056614485840714357434960674649113'),
         -r('0.757391197830719125377033825465364239')],
        [r('1.26198488743988016191264235350272064') * 1j,
         r('1.28526303966470991081119212876082963') * 1j,
         r('0.946914550864355501738777880064324816'),
         r('1.12712800022540204589739296431967767')],
    ])
    change_of_cycles = mp.matrix([
        [0, 0, 0, -1],
        [0, 0, -1, 0],
        [0, -1, 0, 0],
        [-1, 0, 0, 0],
    ])
    relative_error = (
        mp.norm(periods * change_of_cycles - pari_periods)
        / mp.norm(pari_periods))
    assert relative_error < mp.mpf('1e-28')


@pytest.mark.parametrize("coefficients", [
    [0, 4, 0, -5, 0, 1],
    [-48, -4, 64, 5, -17, -1, 1],
])
def test_hyperelliptic_complex_method_matches_real(coefficients):
    # The polygonal continuation specializes to the established real-axis
    # construction, including its odd- and even-degree cycle conventions.
    mp.dps = 25
    real_periods = hyperelliptic_periods(coefficients, method="real")
    complex_periods = hyperelliptic_periods(coefficients, method="complex")
    for real_matrix, complex_matrix in zip(real_periods, complex_periods):
        assert mp.norm(real_matrix - complex_matrix) < mp.mpf('1e-23')


def test_hyperelliptic_complex_sage_oracle():
    # SageMath 10.8, Curve(...).riemann_surface(prec=200,
    # integration_method="rigorous").period_matrix(), for
    # P=(x^2+1)*((x-2)^2+1)*(x-4). Sage integrates x^k dx/(2y), so its
    # matrix is doubled here. The exact symplectic matrix relates Sage's
    # independently constructed cycles to our complex polygonal basis.
    mp.dps = 30
    coefficients = [-20, 21, -28, 22, -8, 1]
    omega, omega_prime, unused_tau = hyperelliptic_periods(coefficients)
    periods = mp.matrix(2, 4)
    periods[:, :2] = omega
    periods[:, 2:] = omega_prime
    c = mp.mpc
    sage_periods = mp.matrix([
        [c('1.56384671072875491960941104798133352',
           '-1.45055102586162674475302640719115732'),
         c('0.931285985364503503267561065416687167',
           '-1.00889336426389769076190379066101802'),
         c('1.26512145072850283268369996512929270'),
         c('-0.333835465363999329416138899712605535',
           '1.00889336426389769076190379066101802')],
        [c('1.76833358466292630035775536790038267',
           '-1.20041069205898473031384430225626537'),
         c('1.93446369874488951042502688680763613',
           '-3.74579067488946915960599882696213800'),
         c('-0.332260228163926420134543037814506931'),
         c('2.26672392690881593055956992462214307',
           '3.74579067488946915960599882696213800')],
    ])
    change_of_cycles = mp.matrix([
        [1, 0, 1, -1],
        [-1, -1, 0, 0],
        [-1, 0, 0, 0],
        [1, 1, 0, -1],
    ])
    relative_error = (
        mp.norm(periods * change_of_cycles - sage_periods)
        / mp.norm(sage_periods))
    assert relative_error < mp.mpf('1e-28')


def test_hyperelliptic_complex_bernatska_genus_four():
    # Bernatska (2026), Examples 1 and 1a. This starts with her complex curve,
    # constructs all period data, and reproduces the published P-functions.
    # Her first-kind coordinates are minus one half of ours in reverse order,
    # so second and third derivatives acquire factors 4 and -8 respectively.
    mp.dps = 25
    c = mp.mpc
    coefficients = [
        c(-101860560, 245519280), c(-131012592, 28208616),
        c(-14219032, -29444932), c(4126332, -3930980),
        c(324058, 455846), c(-79138, 82462), c(-7585, 826),
        c(217, -288), c(39, -10), 1,
    ]
    omega, tau, kappa, characteristic = hyperelliptic_kleinian_data(
        coefficients)
    reverse = mp.matrix([
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
    ])
    bernatska_u = mp.matrix([
        c(-1181275, 204397) / 1000000,
        c(736446, -379656) / 10000000,
        c(-480803, 259696) / 100000000,
        c(6105810, -580815) / 10000000000,
    ])
    u = -2 * reverse * bernatska_u
    values = kleinian_p(
        u, omega, tau, kappa,
        ((3, 3), (3, 2), (3, 3, 3), (3, 3, 2)), characteristic)
    converted = (
        4 * values[0], 4 * values[1],
        -8 * values[2], -8 * values[3],
    )
    expected = (
        c(-5, 4), c(44, 46),
        c(105080464, -182470785) / 1000000,
        c(-106944505, -1805409841) / 1000000,
    )
    for value, reference in zip(converted, expected):
        assert abs(value - reference) < mp.mpf('0.1')


def test_hyperelliptic_second_kind_bernatska_genus_four():
    # J. Bernatska, "Computation of P-Functions on Plane Algebraic Curves",
    # J. Exp. Math. 2(1) (2026), Example 2. Her first-kind basis is minus one
    # half of ours in reverse order, while her associated second-kind basis is
    # minus twice ours in reverse order. Thus kappa_B = 4*R*kappa*R.
    mp.dps = 25
    coefficients = [
        -39916800, 54907920, -11079084, -4495768, 506395,
        82441, -4602, -514, 11, 1,
    ]
    unused_omega, unused_tau, kappa, characteristic = (
        hyperelliptic_kleinian_data(coefficients))
    half = 0.5
    assert characteristic == (
        (half, half, half, half), (0, half, 0, half))
    reverse = mp.matrix([
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
    ])
    bernatska_kappa = 4 * reverse * kappa * reverse
    expected = mp.matrix([
        [-13.123159, 129.285113, 1107.820797, -1910.386399],
        [129.285113, 1362.173530, -26772.601447, 34575.690532],
        [1107.820797, -26772.601447, -519356.757226, 988034.553637],
        [-1910.386399, 34575.690532, 988034.553637, -5074619.889795],
    ])
    relative_error = mp.norm(bernatska_kappa - expected) / mp.norm(expected)
    assert relative_error < mp.mpf('1e-10')


def test_hyperelliptic_periods_pari_oracle():
    # PARI/GP 2.18.1 alpha, hyperellperiods(P, 1), at 70 decimal digits for
    # P=(x+3)(x+1)(x-1/2)(x-2)(x-5). PARI interleaves its symplectic pairs
    # and uses the opposite intersection orientation. After regrouping its
    # columns, the exact anti-symplectic matrix below maps our cycles to its
    # independently constructed cycles.
    mp.dps = 30
    omega, omega_prime, unused_tau = hyperelliptic_periods(
        [-15, 20.5, 26.5, -13.5, -3.5, 1])
    periods = mp.matrix(2, 4)
    periods[:, 0:2] = omega
    periods[:, 2:4] = omega_prime
    r = mp.mpf
    pari_periods = mp.matrix([
        [r('1.08130011176124021123551600052774032'),
         -r('0.367824197708891477585526017213356091') * 1j,
         -r('1.13551151666134456085978436628177736') * 1j,
         r('0.272693341425621752901536388960293600')],
        [r('1.28638852879054877368687826660176149'),
         r('2.60856570108765335185849523088651511') * 1j,
         r('0.238892490675903278038207391105101907') * 1j,
         r('2.73958313544434386763423081630770745')],
    ])
    change_of_cycles = mp.matrix([
        [0, 0, 0, -1],
        [-1, 0, 0, -1],
        [0, -1, -1, 0],
        [0, 0, 1, 0],
    ])
    relative_error = (
        mp.norm(periods * change_of_cycles - pari_periods)
        / mp.norm(pari_periods))
    assert relative_error < mp.mpf('1e-28')


def test_hyperelliptic_periods_sage_genus_three_oracle():
    # SageMath 10.8, Curve(...).riemann_surface(prec=233), followed by
    # period_matrix() with rigorous integration. This is the independent
    # arbitrary-precision period construction used in the nbruin/RiemannTheta
    # README. Sage uses x^k dx/(2y), so its period matrix is doubled below.
    # Its automatically selected cycles are related to ours by the exact
    # symplectic matrix recorded here.
    mp.dps = 30
    r = mp.mpf
    coefficients = [
        r(693) / 25, -r(60021) / 500, -r(131327) / 1000,
        r(9223) / 50, r(47179) / 1000, -r(2969) / 100,
        -r(31) / 10, 1,
    ]
    omega, omega_prime, unused_tau = hyperelliptic_periods(coefficients)
    periods = mp.matrix(3, 6)
    periods[:, 0:3] = omega
    periods[:, 3:6] = omega_prime
    sage_periods = 2 * mp.matrix([
        [-r('0.174993997276248518508584616329222736') * 1j,
         -r('0.057436756315399741218100606662656419') * 1j,
         -r('0.027146339729186879431698083188509171') * 1j,
         r('0.212402081711399673009900834734308171'),
         r('0.153313795498446904641498096851241734')
         - r('0.057436756315399741218100606662656419') * 1j,
         -r('0.140489291807928670412981638571991389')],
        [r('0.247701164039406745020671820107167095') * 1j,
         -r('0.227082395647476355817413167012463233') * 1j,
         r('0.146917339870334651206688748274795459') * 1j,
         -r('0.047499548305822843511766343968151466'),
         r('0.276589924190216037167445078048417100')
         - r('0.227082395647476355817413167012463233') * 1j,
         -r('0.174290831816759871983707261022848997')],
        [-r('0.419674170983378819389464415661593661') * 1j,
         -r('0.951630024105941286510015772184094522') * 1j,
         -r('1.272752059842803424099118457708256402') * 1j,
         r('0.032282936012518007258545325218826891'),
         r('0.563616070467696045418960437098494901')
         - r('0.951630024105941286510015772184094522') * 1j,
         r('0.670736614919071603056268289190659540')],
    ])
    change_of_cycles = mp.matrix([
        [0, 0, 0, 0, 0, -1],
        [0, 0, 0, 1, 0, -1],
        [0, 0, 0, 0, -1, 0],
        [1, 0, 1, 0, 0, 0],
        [-1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0],
    ])
    relative_error = (
        mp.norm(periods * change_of_cycles - sage_periods)
        / mp.norm(sage_periods))
    assert relative_error < mp.mpf('1e-28')


@pytest.mark.parametrize("coefficients, message", [
    ([], "leading coefficient"),
    ([1, 2, 3, 0], "leading coefficient"),
    ([1, 0, 1], "at least 3"),
    ([1, object(), 0, 1], "sequence of numbers"),
    ([1, mp.inf, 0, 1], "finite numbers"),
    ([0, 0, -1, 1], "distinct roots"),
])
def test_hyperelliptic_periods_validation(coefficients, message):
    mp.dps = 20
    with pytest.raises(ValueError, match=message):
        hyperelliptic_periods(coefficients)
    with pytest.raises(ValueError, match="method"):
        hyperelliptic_periods([0, -1, 0, 1], method="unknown")


def test_hyperelliptic_real_method_rejects_complex_data():
    with pytest.raises(ValueError, match="finite real"):
        hyperelliptic_periods([1, 1j, 0, 1], method="real")
    with pytest.raises(ValueError, match="only real roots"):
        hyperelliptic_periods([0, 1, 0, 1], method="real")


def test_hyperelliptic_periods_rejects_inconsistent_integrals(monkeypatch):
    intervals = {
        0: (mp.mpf('0.5'), 0),
        1: (0, 0),
        2: (0, mp.mpf('0.5')),
        3: (mp.mpf('0.5'), 0),
    }
    monkeypatch.setattr(
        hyperelliptic, "_real_branch_integrals",
        lambda ctx, roots, leading, interval, genus: intervals[interval])
    with pytest.raises(ValueError, match="symmetric period matrix"):
        hyperelliptic_periods([0, 4, 0, -5, 0, 1])


def test_hyperelliptic_periods_rejects_inconsistent_legendre_data():
    zero = mp.zeros(1)
    with pytest.raises(ValueError, match="generalized Legendre"):
        hyperelliptic._validate_legendre_relation(
            mp, mp.eye(1), zero, zero, zero, +mp.eps)


def test_hyperelliptic_rejects_inconsistent_sheet_transport(monkeypatch):
    monkeypatch.setattr(
        hyperelliptic, "_complex_branch_data",
        lambda ctx, roots, leading, interval, count: ((ctx.one,), 1, 1))
    with pytest.raises(ValueError, match="square-root sheet"):
        hyperelliptic._complex_branch_integrals(mp, (0, 1, 2), 1, 1)
