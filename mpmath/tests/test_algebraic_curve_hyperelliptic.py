import pytest

from mpmath import mp
from mpmath.curves._hyperelliptic import (
    _hyperelliptic_abel_map, _hyperelliptic_periods,
)
from mpmath.curves._hyperelliptic import integration as hyperelliptic_integration
from mpmath.curves._hyperelliptic import jacobian as hyperelliptic_jacobian
from mpmath.curves._hyperelliptic import model as hyperelliptic_model


def hyperelliptic_periods(coefficients, **kwargs):
    return _hyperelliptic_periods(mp, coefficients, **kwargs)


def hyperelliptic_abel_map(coefficients, target, **kwargs):
    return _hyperelliptic_abel_map(mp, coefficients, target, **kwargs)


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

def test_hyperelliptic_periods_real_oval_orientations():
    # The square-root sheet is continued from the rightmost interval rather
    # than reset to the positive principal root on every real oval. For this
    # odd-degree genus-two curve, dx/y is therefore positive on the first
    # a-cut and negative on the second one.
    mp.dps = 30
    omega, unused_omega_prime, unused_tau = hyperelliptic_periods(
        [0, 4, 0, -5, 0, 1], method="real")
    assert abs(mp.im(omega[0, 0])) < mp.mpf('1e-28')
    assert abs(mp.im(omega[0, 1])) < mp.mpf('1e-28')
    assert mp.re(omega[0, 0]) > 0
    assert mp.re(omega[0, 1]) < 0

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
    omega1 = omega[0, 0]
    omega2 = omega_prime[0, 0]
    unused_q, unused_scale, negative_kappa = mp._weierzeta_data(
        omega1, omega2)
    assert mp.almosteq(tau[0, 0], omega2 / omega1)
    assert mp.almosteq(kappa[0, 0], -negative_kappa)
    assert mp.almosteq(eta[0, 0], kappa[0, 0] * omega[0, 0])
    assert mp.almosteq(
        omega[0, 0] * eta_prime[0, 0]
        - omega_prime[0, 0] * eta[0, 0],
        -mp.pi * 1j / 2)

def test_hyperelliptic_second_kind_genus_two_legendre_relation():
    # Buchstaber, Enolskii & Leykin (1997), Lemma 1.1, in its half-period
    # convention.
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
    expected = -mp.pi * 1j * symplectic / 2
    assert mp.norm(
        periods * symplectic * periods.T - expected) < mp.mpf('1e-27')

def test_hyperelliptic_abel_map_second_kind_differential():
    # Differentiating the incomplete integral must recover BEL (1997),
    # equation (1.3). This genus-two case also exercises the higher-order
    # finite part at the odd-degree point at infinity.
    mp.dps = 35
    coefficients = [0, 16, 0, -20, 0, 4]
    second_coefficients = coefficients + [0]
    x = mp.mpf(3)
    step = mp.mpf('1e-12')

    def point(value):
        polynomial = mp.fsum(
            coefficient * value ** degree
            for degree, coefficient in enumerate(coefficients))
        return value, mp.sqrt(polynomial)

    unused_left, second_left = hyperelliptic_abel_map(
        coefficients, point(x - step), second_kind=True)
    unused_right, second_right = hyperelliptic_abel_map(
        coefficients, point(x + step), second_kind=True)
    y = point(x)[1]
    for row in range(2):
        j = row + 1
        polynomial = mp.fsum(
            (power + 1 - j) * second_coefficients[power + 1 + j]
            * x ** power / 4
            for power in range(j, 6 - j))
        derivative = (second_right[row] - second_left[row]) / (2 * step)
        assert abs(derivative - polynomial / y) < mp.mpf('1e-22')

def test_hyperelliptic_abel_map_second_kind_even_degree():
    # With even degree, both first- and second-kind maps use the finite branch
    # point e0 as base point. The quartic therefore needs no finite-part
    # regularization, and both returned vectors vanish there.
    mp.dps = 30
    coefficients = [24, 14, -13, -2, 1]
    image, second = hyperelliptic_abel_map(
        coefficients, (-3, 0), second_kind=True)
    assert mp.norm(image) == mp.norm(second) == 0

    x = mp.mpf(5)
    y = mp.sqrt(mp.fsum(
        coefficient * x ** degree
        for degree, coefficient in enumerate(coefficients)))
    image, second = hyperelliptic_abel_map(
        coefficients, (x, y), second_kind=True, reduce=True)
    assert all(mp.isfinite(value) for value in image)
    assert all(mp.isfinite(value) for value in second)

    step = mp.mpf('1e-10')

    def second_value(value):
        ordinate = mp.sqrt(mp.fsum(
            coefficient * value ** degree
            for degree, coefficient in enumerate(coefficients)))
        unused_image, result = hyperelliptic_abel_map(
            coefficients, (value, ordinate), second_kind=True)
        return result[0]

    derivative = (
        second_value(x + step) - second_value(x - step)) / (2 * step)
    expected = (coefficients[3] * x / 4
                + coefficients[4] * x ** 2 / 2) / y
    assert abs(derivative - expected) < mp.mpf('1e-18')

def test_hyperelliptic_abel_map_sage_oracle():
    # SageMath 10.8 RiemannSurface.abel_jacobi at 130 bits, on the two
    # sheets above x=2 of y^2=4*x^3-4*x. Subtracting the two images removes
    # Sage's base point. Sage integrates dx/(2*y), so its sheet difference
    # 0.58408284167715170669284916892566789240 is doubled here.
    mp.dps = 35
    coefficients = [0, -4, 0, 4]
    x = mp.mpf(2)
    y = mp.sqrt(24)
    difference = (
        hyperelliptic_abel_map(coefficients, (x, y))
        - hyperelliptic_abel_map(coefficients, (x, -y)))
    positive_image = hyperelliptic_abel_map(coefficients, (x, y))
    assert mp.almosteq(
        positive_image[0], -mp.elliprf(x + 1, x, x - 1))
    expected = 2 * mp.mpf(
        '0.58408284167715170669284916892566789240')
    assert abs(difference[0] - expected) < mp.mpf('1e-34')

@pytest.mark.parametrize("coefficients, branch", [
    ([0, 16, 0, -20, 0, 4], 1),
    ([6, 1, -7, -1, 1], 1),
])
def test_hyperelliptic_abel_map_branch_point_torsion(coefficients, branch):
    # Twice a branch point minus twice the Abel-map base point is principal,
    # hence every branch-point image is two-torsion in the Jacobian.
    mp.dps = 25
    roots = mp.polyroots(coefficients)
    x = sorted(roots, key=lambda root: (mp.re(root), mp.im(root)))[branch]
    image = hyperelliptic_abel_map(
        coefficients, ((x, 0), (x, 0)), reduce=True)
    assert mp.norm(image) < mp.mpf('1e-22')

def test_hyperelliptic_abel_map_complex_curve_involution():
    # Abel's theorem applies without a real branch-point ordering as well:
    # P plus its hyperelliptic involution maps to zero modulo full periods.
    mp.dps = 25
    coefficients = [-20, 21, -28, 22, -8, 1]
    x = mp.mpc('0.3', '0.2')
    y = mp.sqrt(sum(
        coefficient * x ** degree
        for degree, coefficient in enumerate(coefficients)))
    image = hyperelliptic_abel_map(
        coefficients, ((x, y), (x, -y)), reduce=True)
    assert mp.norm(image) < mp.mpf('1e-22')

@pytest.mark.parametrize("target, message", [
    (1, "affine point"),
    ([1], "pair"),
    (((1,),), "pair"),
    (((1, 2), 3), "pair"),
    (((1, object()),), "numbers"),
    ((mp.inf, 0), "finite"),
    (((0, 0), (mp.inf, 0)), "finite"),
])
def test_hyperelliptic_abel_map_target_validation(target, message):
    with pytest.raises(ValueError, match=message):
        hyperelliptic_abel_map([0, -4, 0, 4], target)

def test_hyperelliptic_abel_map_rejects_point_off_curve():
    with pytest.raises(ValueError, match=r"y\*\*2 = P\(x\)"):
        hyperelliptic_abel_map([0, -4, 0, 4], (1, 1))

def test_hyperelliptic_abel_map_internal_failures(monkeypatch):
    with pytest.raises(ValueError, match="square-root sheet"):
        hyperelliptic_integration._branch_target_integrals(
            mp, (0, 1, 2), 1, 0, 3, 1, 1, +mp.eps)

    zero = mp.zeros(1, 1)
    with pytest.raises(ValueError, match="full period lattice"):
        hyperelliptic_jacobian._abel_lattice_shift(
            mp, zero, zero, zero, +mp.eps)

    monkeypatch.setattr(
        mp, "lu_solve", lambda matrix, vector: mp.matrix([0, 0]))
    with pytest.raises(ValueError, match="full period lattice"):
        hyperelliptic_jacobian._abel_lattice_shift(
            mp, mp.matrix([1]), mp.matrix([[1]]), mp.matrix([[1j]]),
            +mp.eps)

def test_hyperelliptic_infinity_ray_avoids_roots(monkeypatch):
    direction = hyperelliptic_model._infinity_direction(mp, (3, 1, 2))
    assert not mp.almosteq(direction, 1)

    monkeypatch.setattr(mp, "exp", lambda value: mp.one)
    with pytest.raises(ValueError, match="root-free path"):
        hyperelliptic_model._infinity_direction(mp, (3, 1, 2))

def test_hyperelliptic_endpoint_limits(monkeypatch):
    # The quadrature implementation does not ordinarily sample exact
    # endpoints, but these removable-limit branches support backends that do.
    monkeypatch.setattr(
        mp, "quad", lambda function, interval: (
            function(0) + function(mp.mpf('0.5')) + function(1)))
    values = hyperelliptic_integration._infinity_branch_integrals(
        mp, (-2, -1, 0, 1, 2), 1, 2)
    assert all(mp.isfinite(value) for value in values)
    monkeypatch.setattr(
        mp, "quad", lambda function, interval: function(mp.inf))
    values = hyperelliptic_integration._infinity_second_kind_integrals(
        mp, (0, -4, 0, 4, 0), (-1, 0, 1), 1)
    assert all(mp.isfinite(value) for value in values)
    with pytest.raises(ValueError, match="branch-point path"):
        hyperelliptic_model._admissible_branch_vertex(mp, 0, (), +mp.eps)

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
    # PARI returns complete periods; divide them by two for this interface.
    pari_periods = mp.matrix([
        [r('0.464901227905802907214915155765884188') * 1j,
         -r('0.531986563561318428544478628778286555') * 1j,
         r('0.645056614485840714357434960674649113'),
         -r('0.757391197830719125377033825465364239')],
        [r('1.26198488743988016191264235350272064') * 1j,
         r('1.28526303966470991081119212876082963') * 1j,
         r('0.946914550864355501738777880064324816'),
         r('1.12712800022540204589739296431967767')],
    ]) / 2
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
    # complete cycle integrals equal our half-periods. The exact symplectic
    # matrix relates Sage's cycles to our complex polygonal basis.
    mp.dps = 30
    coefficients = [-20, 21, -28, 22, -8, 1]
    omega, omega_prime, unused_tau = hyperelliptic_periods(coefficients)
    periods = mp.matrix(2, 4)
    periods[:, :2] = omega
    periods[:, 2:] = omega_prime
    c = mp.mpc
    # These literals were originally recorded as twice Sage's output.
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
    ]) / 2
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
    # PARI returns complete periods; divide them by two for this interface.
    pari_periods = mp.matrix([
        [r('1.08130011176124021123551600052774032'),
         -r('0.367824197708891477585526017213356091') * 1j,
         -r('1.13551151666134456085978436628177736') * 1j,
         r('0.272693341425621752901536388960293600')],
        [r('1.28638852879054877368687826660176149'),
         r('2.60856570108765335185849523088651511') * 1j,
         r('0.238892490675903278038207391105101907') * 1j,
         r('2.73958313544434386763423081630770745')],
    ]) / 2
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
    # README. Sage uses x^k dx/(2y), so its complete cycle integrals equal our
    # half-periods. Its cycles are related to ours by the exact symplectic
    # matrix recorded here.
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
    sage_periods = mp.matrix([
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
        hyperelliptic_integration, "_real_branch_integrals",
        lambda ctx, roots, leading, interval, genus: intervals[interval])
    with pytest.raises(ValueError, match="symmetric period matrix"):
        hyperelliptic_periods([0, 4, 0, -5, 0, 1])

def test_hyperelliptic_periods_rejects_inconsistent_legendre_data():
    zero = mp.zeros(1)
    with pytest.raises(ValueError, match="generalized Legendre"):
        hyperelliptic_jacobian._validate_legendre_relation(
            mp, mp.eye(1), zero, zero, zero, +mp.eps)

def test_hyperelliptic_rejects_inconsistent_sheet_transport(monkeypatch):
    monkeypatch.setattr(
        hyperelliptic_integration, "_complex_branch_data",
        lambda ctx, roots, leading, interval, count: ((ctx.one,), 1, 1))
    with pytest.raises(ValueError, match="square-root sheet"):
        hyperelliptic_integration._complex_branch_integrals(
            mp, (0, 1, 2), 1, 1)
