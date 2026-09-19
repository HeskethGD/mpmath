import pytest

from mpmath import (
    hyperelliptic_abel_map, hyperelliptic_data, hyperelliptic_periods,
    kleinian_baker_akhiezer, kleinian_p, kleinian_sigma, kleinian_sigma_jet,
    kleinian_zeta, mp, weierp, weierpprime, weiersigma, weierzeta,
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


def test_hyperelliptic_data_genus_one():
    # In genus one, Bernatska's sum of branch-point characteristics gives the
    # classical odd characteristic [1/2, 1/2]. The resulting logarithmic
    # derivatives and the Schur-normalized sigma agree with the conventional
    # Weierstrass functions.
    mp.dps = 30
    omega, tau, kappa, characteristic = hyperelliptic_data(
        [0, -1, 0, 1])
    assert characteristic == ((0.5,), (0.5,))
    omega1 = omega[0, 0]
    omega2 = (omega * tau)[0, 0]
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


def test_hyperelliptic_abel_map_genus_one_inversion():
    # The classical Weierstrass parametrization is x=wp(u), y=wp'(u) for
    # y^2=4*x^3-4*x. Its inverse checks the Abel map, its sheet convention,
    # and compatibility with the Kleinian functions independently of the
    # path chosen by the numerical integral.
    mp.dps = 30
    coefficients = [0, -4, 0, 4]
    omega, tau, kappa, characteristic = hyperelliptic_data(coefficients)
    omega1 = omega[0, 0]
    omega2 = (omega * tau)[0, 0]
    for u in (mp.mpf('0.3'), mp.mpc('0.3', '0.07')):
        x = weierp(u, omega1=omega1, omega2=omega2)
        y = weierpprime(u, omega1=omega1, omega2=omega2)
        image = hyperelliptic_abel_map(coefficients, (x, y))
        reduced = hyperelliptic_abel_map(
            coefficients, (x, y), reduce=True)
        assert mp.almosteq(reduced[0], u)
        assert abs(kleinian_p(
            image, omega, tau, kappa, (0, 0), characteristic) - x) < (
                mp.mpf('1e-27') * max(1, abs(x)))
        assert abs(kleinian_p(
            image, omega, tau, kappa, (0, 0, 0), characteristic) - y) < (
                mp.mpf('1e-27') * max(1, abs(y)))

        # P + its hyperelliptic involution is a principal divisor relative
        # to two copies of the point at infinity, so its Abel image is zero
        # modulo the period lattice (Abel's theorem).
        involution = hyperelliptic_abel_map(
            coefficients, ((x, y), (x, -y)), reduce=True)
        assert mp.norm(involution) < mp.mpf('1e-28')
    assert mp.norm(hyperelliptic_abel_map(coefficients, ())) == 0


def test_hyperelliptic_abel_map_second_kind_genus_one():
    # BEL (1997), equation (1.3), gives dr=x*dx/y on
    # y**2=4*x**3-4*x. Under x=wp(u), y=wp'(u), its finite-part integral
    # from infinity is -zeta(u). This also pins the regularization constant.
    mp.dps = 35
    coefficients = [0, -4, 0, 4]
    omega, tau, unused_kappa, unused_characteristic = hyperelliptic_data(
        coefficients)
    omega1 = omega[0, 0]
    omega2 = (omega * tau)[0, 0]
    u = mp.mpc('0.3', '0.07')
    x = weierp(u, omega1=omega1, omega2=omega2)
    y = weierpprime(u, omega1=omega1, omega2=omega2)
    image, second = hyperelliptic_abel_map(
        coefficients, (x, y), second_kind=True, reduce=True)
    assert mp.almosteq(image[0], u)
    assert mp.almosteq(
        second[0], -weierzeta(u, omega1=omega1, omega2=omega2))

    # At this branch point, fundamental-cell reduction changes the path by
    # one a-cycle. The zeta identity verifies the coupled eta correction.
    for reduce in (False, True):
        image, second = hyperelliptic_abel_map(
            coefficients, (1, 0), second_kind=True, reduce=reduce)
        assert mp.almosteq(
            second[0],
            -weierzeta(image[0], omega1=omega1, omega2=omega2))

    image, second = hyperelliptic_abel_map(
        coefficients, (), second_kind=True)
    assert mp.norm(image) == mp.norm(second) == 0


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


def test_kleinian_baker_akhiezer_genus_one_schrodinger():
    # The genus-one specialization of CEEK (2000), equations (3.21)-(3.22),
    # is the classical one-gap Lame equation. The factor depending only on
    # the spectral point cancels from this differential equation.
    mp.dps = 30
    coefficients = [0, -4, 0, 4]
    omega, tau, kappa, characteristic = hyperelliptic_data(coefficients)
    x = mp.mpf(2)
    y = mp.sqrt(4 * x ** 3 - 4 * x)
    abel, second = hyperelliptic_abel_map(
        coefficients, (x, y), second_kind=True)

    def baker(argument):
        return kleinian_baker_akhiezer(
            [argument], abel, second, omega, tau, kappa, characteristic)

    u = mp.mpf('0.3')
    value = baker(u)
    potential = kleinian_p(
        [u], omega, tau, kappa, (0, 0), characteristic)
    eigenvalue = (mp.diff(baker, u, 2) - 2 * potential * value) / value
    assert mp.almosteq(eigenvalue, x)


def test_kleinian_baker_akhiezer_genus_two_schrodinger():
    # CEEK (2000), equation (3.22), states
    # (d_2**2-2*wp_22) Psi = (lambda+alpha_4/4) Psi. Here alpha_4=0.
    # This tests the sigma quotient, the finite-part second-kind vector and
    # the conversion from CEEK's positive eta convention together.
    mp.dps = 30
    coefficients = [0, 16, 0, -20, 0, 4]
    omega, tau, kappa, characteristic = hyperelliptic_data(coefficients)
    x = mp.mpf(3)
    y = mp.sqrt(mp.fsum(
        coefficient * x ** degree
        for degree, coefficient in enumerate(coefficients)))
    abel, second = hyperelliptic_abel_map(
        coefficients, (x, y), second_kind=True)
    u1 = mp.mpf('0.13')

    def baker(u2):
        return kleinian_baker_akhiezer(
            [u1, u2], abel, second, omega, tau, kappa, characteristic)

    u2 = mp.mpf('0.27')
    value = baker(u2)
    potential = kleinian_p(
        [u1, u2], omega, tau, kappa, (1, 1), characteristic)
    eigenvalue = (mp.diff(baker, u2, 2) - 2 * potential * value) / value
    assert abs(eigenvalue - x) < mp.mpf('1e-27')


def test_kleinian_baker_akhiezer_genus_two_product():
    # Braden, Enolskii & Hone (2005), equations (3.3) and (3.8), normalize
    # the two sheet-related Baker functions by sigma_2(A(P)) and give their
    # product as the genus-two Bolza polynomial.
    mp.dps = 30
    coefficients = [0, 16, 0, -20, 0, 4]
    omega, tau, kappa, characteristic = hyperelliptic_data(coefficients)
    x = mp.mpf(3)
    y = mp.sqrt(mp.fsum(
        coefficient * x ** degree
        for degree, coefficient in enumerate(coefficients)))
    u = [mp.mpf('0.13'), mp.mpf('0.27')]
    values = []
    for ordinate in (y, -y):
        abel, second = hyperelliptic_abel_map(
            coefficients, (x, ordinate), second_kind=True)
        values.append(kleinian_baker_akhiezer(
            u, abel, second, omega, tau, kappa, characteristic))
    wp_22, wp_12 = kleinian_p(
        u, omega, tau, kappa, [(1, 1), (0, 1)], characteristic)
    bolza = x ** 2 - wp_22 * x - wp_12
    assert abs(values[0] * values[1] - bolza) < mp.mpf('1e-27')


def test_sigma_stratum_derivatives_genus_three_addition():
    # Gibbons, Matsutani & Onishi (2013), Proposition 7.9, gives the
    # arbitrary-genus one-point-stratum addition formula. In genus three,
    # sigma_natural1 = sigma_2 and sigma_natural2 = sigma_3. With mpmath's
    # argument ordering its right side is x_v-x_u. This checks both the
    # sigma-sharp derivative selected by the BA normalization and its scale.
    mp.dps = 25
    coefficients = [0, -144, 0, 196, 0, -56, 0, 4]
    omega, tau, kappa, characteristic = hyperelliptic_data(coefficients)

    def image(x):
        y = mp.sqrt(mp.fsum(
            coefficient * x ** degree
            for degree, coefficient in enumerate(coefficients)))
        return hyperelliptic_abel_map(coefficients, (x, y))

    x_u = mp.mpf(4)
    x_v = mp.mpf(5)
    u = image(x_u)
    v = image(x_v)

    def derivative(argument, index):
        jet = kleinian_sigma_jet(
            argument, omega, tau, kappa, 1, characteristic,
            normalization="hyperelliptic")
        return jet[index]

    plus = [u[index] + v[index] for index in range(3)]
    minus = [u[index] - v[index] for index in range(3)]
    sigma_natural2_product = (
        derivative(plus, (0, 0, 1))
        * derivative(minus, (0, 0, 1)))
    sigma_sharp_product = (
        derivative(u, (0, 1, 0)) ** 2
        * derivative(v, (0, 1, 0)) ** 2)
    result = sigma_natural2_product / sigma_sharp_product
    assert abs(result - (x_v - x_u)) < mp.mpf('1e-22')


def test_kleinian_baker_akhiezer_vector_validation():
    omega = [[1]]
    tau = [[1j]]
    kappa = [[0]]
    with pytest.raises(ValueError, match="abel must have length 1"):
        kleinian_baker_akhiezer([0.2], [0.1, 0.2], [0.3],
                                omega, tau, kappa)
    with pytest.raises(ValueError, match="second_kind must have length 1"):
        kleinian_baker_akhiezer([0.2], [0.1], [0.2, 0.3],
                                omega, tau, kappa)


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


def test_hyperelliptic_abel_map_genus_two_jacobi_inversion():
    # BEL (1997), equation (3.10), gives x1+x2=wp_22 and
    # x1*x2=-wp_12. Equation (3.11) gives
    # y_k=wp_222*x_k+wp_122 in the present y^2=P(x) convention.
    mp.dps = 30
    coefficients = [0, 16, 0, -20, 0, 4]
    omega, tau, kappa, characteristic = hyperelliptic_data(coefficients)
    u = mp.matrix([mp.mpf('0.13'), mp.mpf('0.27')])
    indices = ((1, 1), (0, 1), (1, 1, 1), (0, 1, 1))
    p22, p12, p222, p122 = kleinian_p(
        u, omega, tau, kappa, indices, characteristic)
    roots = mp.polyroots([-p12, -p22, 1], maxsteps=200)
    divisor = tuple((x, p222 * x + p122) for x in roots)

    image = hyperelliptic_abel_map(coefficients, divisor)
    reduced = hyperelliptic_abel_map(
        coefficients, divisor, reduce=True)
    assert mp.norm(reduced - u) < mp.mpf('1e-27')
    recovered = kleinian_p(
        image, omega, tau, kappa, indices, characteristic)
    for value, expected in zip(recovered, (p22, p12, p222, p122)):
        assert abs(value - expected) < (
            mp.mpf('1e-26') * max(1, abs(expected)))

    # An Abel map is additive on divisors before lattice reduction.
    point_images = [
        hyperelliptic_abel_map(coefficients, point) for point in divisor]
    assert mp.norm(image - sum(point_images, mp.zeros(2, 1))) < mp.mpf('1e-28')
    complex_image = hyperelliptic_abel_map(
        coefficients, divisor, method="complex")
    assert mp.norm(complex_image - image) < mp.mpf('1e-28')


def test_hyperelliptic_abel_map_complex_jacobi_inversion():
    # The same BEL (1997), equations (3.10)-(3.11), closed loop exercises
    # the polygonal backbone and the path from infinity when the canonical
    # genus-two curve has two conjugate pairs of branch points.
    mp.dps = 30
    coefficients = [80, -44, 80, -40, 0, 4]
    omega, tau, kappa, characteristic = hyperelliptic_data(coefficients)
    u = mp.matrix([mp.mpc('0.13', '0.02'), mp.mpc('0.27', '-0.01')])
    indices = ((1, 1), (0, 1), (1, 1, 1), (0, 1, 1))
    p22, p12, p222, p122 = kleinian_p(
        u, omega, tau, kappa, indices, characteristic)
    roots = mp.polyroots([-p12, -p22, 1], maxsteps=200)
    divisor = tuple((x, p222 * x + p122) for x in roots)
    image = hyperelliptic_abel_map(
        coefficients, divisor, method="complex", reduce=True)
    assert mp.norm(image - u) < mp.mpf('1e-27')


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
        hyperelliptic._branch_target_integrals(
            mp, (0, 1, 2), 1, 0, 3, 1, 1, +mp.eps)

    zero = mp.zeros(1, 1)
    with pytest.raises(ValueError, match="full period lattice"):
        hyperelliptic._abel_lattice_shift(mp, zero, zero, zero, +mp.eps)

    monkeypatch.setattr(
        mp, "lu_solve", lambda matrix, vector: mp.matrix([0, 0]))
    with pytest.raises(ValueError, match="full period lattice"):
        hyperelliptic._abel_lattice_shift(
            mp, mp.matrix([1]), mp.matrix([[1]]), mp.matrix([[1j]]),
            +mp.eps)


def test_hyperelliptic_infinity_ray_avoids_roots(monkeypatch):
    direction = hyperelliptic._infinity_direction(mp, (3, 1, 2))
    assert not mp.almosteq(direction, 1)

    monkeypatch.setattr(mp, "exp", lambda value: mp.one)
    with pytest.raises(ValueError, match="root-free path"):
        hyperelliptic._infinity_direction(mp, (3, 1, 2))


def test_hyperelliptic_endpoint_limits(monkeypatch):
    # The quadrature implementation does not ordinarily sample exact
    # endpoints, but these removable-limit branches support backends that do.
    monkeypatch.setattr(
        mp, "quad", lambda function, interval: (
            function(0) + function(mp.mpf('0.5')) + function(1)))
    values = hyperelliptic._infinity_branch_integrals(
        mp, (-2, -1, 0, 1, 2), 1, 2)
    assert all(mp.isfinite(value) for value in values)
    monkeypatch.setattr(
        mp, "quad", lambda function, interval: function(mp.inf))
    values = hyperelliptic._infinity_second_kind_integrals(
        mp, (0, -4, 0, 4, 0), (-1, 0, 1), 1)
    assert all(mp.isfinite(value) for value in values)
    with pytest.raises(ValueError, match="branch-point path"):
        hyperelliptic._admissible_branch_vertex(mp, 0, (), +mp.eps)


@pytest.mark.parametrize("coefficients", [
    [0, 16, 0, -20, 0, 4],
    [0, -36, 0, 49, 0, -14, 0, 1],
])
def test_hyperelliptic_sigma_schur_normalization(coefficients):
    # Buchstaber, Enolskii & Leykin (1997), Definition 1: the fundamental
    # sigma function starts with delta(u)=det(u[i+j-1]). The coefficient of
    # u_m**m is the sign of the reversing permutation, m=floor((g+1)/2).
    mp.dps = 35
    omega, tau, kappa, characteristic = hyperelliptic_data(
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


def test_hyperelliptic_data_genus_two_periodicity():
    # Eilbeck, Enolskii & Leykin (2000), Definition 3.7, gives full-lattice
    # periodicity. BEL (1997), equation (3.9), gives the genus-two cubic
    # identity checked below. Together they test that periods, kappa and the
    # Riemann characteristic constructed from one curve work coherently.
    mp.dps = 40
    coefficients = [0, 16, 0, -20, 0, 4]
    omega, tau, kappa, characteristic = hyperelliptic_data(
        coefficients)
    half = 0.5
    assert characteristic == ((half, half), (0, half))
    u = mp.matrix([mp.mpc('0.13', '0.02'), mp.mpc('-0.08', '0.01')])
    shifted = (
        u + 2 * omega * mp.matrix([1, -1])
        + 2 * omega * tau * mp.matrix([1, 1])
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


def test_hyperelliptic_christiansen_sigma_expansion():
    # Christiansen, Eilbeck, Enolskii & Kostov (2000), equation (3.3),
    # Proc. R. Soc. Lond. A 456, doi:10.1098/rspa.2000.0612,
    # gives the first terms of the canonically normalized genus-two sigma
    # function. Ordinary differentiation converts its cubic coefficients to
    # sigma_111(0) = alpha_2/4 and sigma_222(0) = -2.
    mp.dps = 35
    coefficients = [0, mp.mpf(1) / 16, -mp.mpf(15) / 16,
                    mp.mpf(35) / 8, -mp.mpf(15) / 2, 4]
    omega, tau, kappa, characteristic = hyperelliptic_data(coefficients)

    def sigma(u1, u2):
        return kleinian_sigma(
            [u1, u2], omega, tau, kappa, characteristic,
            normalization="hyperelliptic")

    assert mp.almosteq(mp.diff(lambda value: sigma(value, 0), 0), 1)
    assert mp.almosteq(
        mp.diff(lambda value: sigma(value, 0), 0, 3),
        coefficients[2] / 4)
    assert mp.almosteq(
        mp.diff(lambda value: sigma(0, value), 0, 3), -2)


def test_hyperelliptic_christiansen_p_identities():
    # Christiansen, Eilbeck, Enolskii & Kostov (2000), equations
    # (3.10)-(3.12) and (3.15)-(3.17), Proc. R. Soc. Lond. A 456,
    # doi:10.1098/rspa.2000.0612. The paper uses one-based indices; this test
    # translates them to the zero-based kleinian_p interface.
    mp.dps = 35
    coefficients = [0, mp.mpf(1) / 16, -mp.mpf(15) / 16,
                    mp.mpf(35) / 8, -mp.mpf(15) / 2, 4]
    omega, tau, kappa, characteristic = hyperelliptic_data(coefficients)
    u = [mp.mpf("0.13"), mp.mpf("0.27")]
    indices = (
        (0, 0), (0, 1), (1, 1), (0, 1, 1), (1, 1, 1),
        (0, 1, 1, 1), (1, 1, 1, 1),
    )
    p11, p12, p22, p122, p222, p1222, p2222 = kleinian_p(
        u, omega, tau, kappa, indices, characteristic)
    alpha1, alpha2, alpha3, alpha4 = coefficients[1:5]

    identity_315 = (
        p222 * p122
        - (4 * p12 * p22 ** 2 + 2 * p12 ** 2 - 2 * p11 * p22
           + alpha4 * p12 * p22 + alpha3 * p12 / 2 + alpha1 / 2)
    )
    identity_316 = (
        p2222
        - (6 * p22 ** 2 + alpha3 / 2 + alpha4 * p22 + 4 * p12)
    )
    identity_317 = (
        p1222 - (6 * p22 * p12 + alpha4 * p12 - 2 * p11)
    )
    scale = max(1, abs(p2222), abs(p1222), abs(p222 * p122))
    assert max(abs(identity_315), abs(identity_316), abs(identity_317)) < (
        100 * mp.eps * scale)

    # Equations (3.10)-(3.12) solve the Jacobi inversion problem: the roots
    # mu_i of lambda**2-p22*lambda-p12 have curve coordinates
    # nu_i = p222*mu_i+p122.
    mus = mp.polyroots([-p12, -p22, 1])
    for mu in mus:
        nu = p222 * mu + p122
        curve_value = mp.fsum(
            coefficient * mu ** degree
            for degree, coefficient in enumerate(coefficients))
        # Recovering mu numerically and substituting it into a quintic loses
        # a few guard digits compared with the direct identities above.
        root_substitution_tolerance = (
            10000 * mp.eps * max(1, abs(curve_value)))
        assert abs(nu ** 2 - curve_value) < root_substitution_tolerance


def test_hyperelliptic_buchstaber_genus_three_identities():
    # Buchstaber, "The Mumford Dynamical System and Hyperelliptic
    # Kleinian Functions" (2024), arXiv:2402.09218, Corollary 8.2.
    # Its weighted coordinates (z1, z3, z5) correspond to our (u3, u2, u1),
    # so wp2, wp4, wp6 and wp3,3 map respectively to wp33, wp23, wp13 and
    # wp22 in one-based Abelian-coordinate notation.
    mp.dps = 35
    # 4*(x+5)*(x+3)*(x+1)*x*(x-1)*(x-2)*(x-6)
    # = 4*(x**7 - 38*x**5 - 24*x**4 + 217*x**3
    #     + 24*x**2 - 180*x).
    coefficients = [0, -720, 96, 868, -96, -152, 0, 4]
    lambda4 = -38
    lambda6 = -24
    omega, tau, kappa, characteristic = hyperelliptic_data(coefficients)
    u = [mp.mpf('0.13'), mp.mpf('0.17'), mp.mpf('0.19')]
    indices = (
        (2, 2), (1, 2), (0, 2), (1, 1),
        (2, 2, 2), (2, 2, 2, 2), (1, 2, 2, 2),
    )
    p2, p4, p6, p33, p2_prime, p2_second, p4_second = kleinian_p(
        u, omega, tau, kappa, indices, characteristic)

    expected_p2_second = 6 * p2 ** 2 + 4 * p4 + 2 * lambda4
    expected_p4_second = 6 * (p2 * p4 + p6) - 2 * p33
    expected_p2_prime_squared = 4 * (
        p2 ** 3 + (p4 + lambda4) * p2 + p33 - p6 + lambda6)
    comparisons = (
        (p2_second, expected_p2_second),
        (p4_second, expected_p4_second),
        (p2_prime ** 2, expected_p2_prime_squared),
    )
    for value, expected in comparisons:
        assert abs(value - expected) < (
            100 * mp.eps * max(1, abs(value), abs(expected)))


def test_hyperelliptic_enolski_genus_two_branch_identities():
    # Enolski, Hackmann, Kagramanova, Kunz & Lammerzahl (2011),
    # J. Geom. Phys. 61, doi:10.1016/j.geomphys.2011.01.001,
    # equations (5.7)-(5.9), (5.13), and (5.21). Its half-period omega and
    # characteristic ordering agree with this interface: epsilon-prime is
    # rtheta's first component and epsilon is its second.
    mp.dps = 35
    # y**2 = 4*x*(x-1)*(x-2)*(x-3)*(x-4).
    coefficients = [0, 96, -200, 140, -40, 4]
    omega, tau, kappa, characteristic = hyperelliptic_data(coefficients)
    half = 0.5
    assert characteristic == ((half, half), (0, half))

    # This is A1+A2 from (5.7), before reduction modulo complete periods.
    epsilon_prime = mp.matrix([1, 0])
    epsilon = mp.matrix([half, 0])
    omega_12 = 2 * omega * (epsilon + tau * epsilon_prime)
    p22, p12 = kleinian_p(
        omega_12, omega, tau, kappa, ((1, 1), (0, 1)),
        characteristic)
    assert mp.almosteq(p22, 1)
    assert mp.almosteq(p12, 0)

    # Equation (5.21) recovers each finite branch point directly on the
    # theta divisor as e_i = -sigma_1(A_i)/sigma_2(A_i). The table entries
    # below are the upper and lower rows of (5.7) and (5.8).
    branch_characteristics = (
        ((half, 0), (0, 0)),
        ((half, 0), (half, 0)),
        ((0, half), (half, 0)),
        ((0, half), (half, half)),
        ((0, 0), (half, half)),
    )
    for branch_point, (epsilon_prime, epsilon) in enumerate(
            branch_characteristics):
        branch_image = 2 * omega * (
            mp.matrix(epsilon) + tau * mp.matrix(epsilon_prime))
        jet = kleinian_sigma_jet(
            branch_image, omega, tau, kappa, 1, characteristic,
            normalization="hyperelliptic")
        derivative_scale = max(abs(jet[(1, 0)]), abs(jet[(0, 1)]))
        assert abs(jet[(0, 0)]) < 100 * mp.eps * derivative_scale
        recovered = -jet[(1, 0)] / jet[(0, 1)]
        assert abs(recovered - branch_point) < (
            1000 * mp.eps * max(1, branch_point))


def test_hyperelliptic_enolski_genus_three_branch_identities():
    # Enolski et al. (2011), equations (6.7), (6.8), (6.19), (6.26), and the
    # explicit curve (6.27), using the same half-period and characteristic
    # conventions as the genus-two test above.
    mp.dps = 35
    # y**2 = 4*x*(x-1)*(x-2)*(x-3)*(x-4)*(x-5)*(x-6).
    coefficients = [0, 2880, -7056, 6496, -2940, 700, -84, 4]
    omega, tau, kappa, characteristic = hyperelliptic_data(coefficients)
    half = 0.5
    assert characteristic == (
        (half, half, half), (half, 0, half))

    # This is A1+A2+A3 from (6.7), before reduction modulo complete periods.
    epsilon_prime = mp.matrix([1, half, 0])
    epsilon = mp.matrix([1, 0, 0])
    omega_123 = 2 * omega * (epsilon + tau * epsilon_prime)
    p33, p23, p13 = kleinian_p(
        omega_123, omega, tau, kappa,
        ((2, 2), (1, 2), (0, 2)), characteristic)
    assert mp.almosteq(p33, 3)
    assert mp.almosteq(p23, -2)
    # The exact product contains the branch point zero. Its numerical value
    # results from cancellation among period and theta-derivative terms.
    assert abs(p13) < 1000 * mp.eps

    # Equation (6.26) gives the same branch-point recovery quotient in genus
    # three. These characteristics are the seven finite entries of (6.7).
    branch_characteristics = (
        ((half, 0, 0), (0, 0, 0)),
        ((half, 0, 0), (half, 0, 0)),
        ((0, half, 0), (half, 0, 0)),
        ((0, half, 0), (half, half, 0)),
        ((0, 0, half), (half, half, 0)),
        ((0, 0, half), (half, half, half)),
        ((0, 0, 0), (half, half, half)),
    )
    for branch_point, (epsilon_prime, epsilon) in enumerate(
            branch_characteristics):
        branch_image = 2 * omega * (
            mp.matrix(epsilon) + tau * mp.matrix(epsilon_prime))
        jet = kleinian_sigma_jet(
            branch_image, omega, tau, kappa, 1, characteristic,
            normalization="hyperelliptic")
        derivative_scale = max(
            abs(jet[(1, 0, 0)]), abs(jet[(0, 1, 0)]))
        assert abs(jet[(0, 0, 0)]) < 100 * mp.eps * derivative_scale
        recovered = -jet[(1, 0, 0)] / jet[(0, 1, 0)]
        assert abs(recovered - branch_point) < (
            1000 * mp.eps * max(1, branch_point))


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
    omega, tau, kappa, characteristic = hyperelliptic_data(
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
    omega, tau, kappa, characteristic = hyperelliptic_data(
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
    # of ours in reverse order. Her full-period, positive-integral convention
    # gives kappa_B = -4*R*kappa*R in terms of our classical kappa.
    mp.dps = 25
    coefficients = [
        -39916800, 54907920, -11079084, -4495768, 506395,
        82441, -4602, -514, 11, 1,
    ]
    unused_omega, unused_tau, kappa, characteristic = (
        hyperelliptic_data(coefficients))
    half = 0.5
    assert characteristic == (
        (half, half, half, half), (0, half, 0, half))
    reverse = mp.matrix([
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
    ])
    bernatska_kappa = -4 * reverse * kappa * reverse
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
