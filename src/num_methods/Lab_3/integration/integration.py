import math

# 1/(x³ + 64) на отрезке [-2, 2].
def f(x):
    return 1 / (x**3 + 64)

A, B = -2.0, 2.0
H1, H2 = 1.0, 0.5  # H1 > H2

def antiderivative(x):
    t = x / 4
    return (math.log(t + 1) / 3 - math.log(t*t - t + 1) / 6
            + math.atan((2*t - 1) / math.sqrt(3)) / math.sqrt(3)) / 16

EXACT = antiderivative(B) - antiderivative(A)


def count_intervals(a, b, h):
    if not (math.isfinite(a) and math.isfinite(b) and math.isfinite(h)) or b <= a or h <= 0:
        raise ValueError('Требуются конечные a < b и положительный шаг.')

    n = round((b - a) / h)
    if n < 1 or not math.isclose(n * h, b - a, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError('Шаг должен укладываться целое число раз на отрезке.')

    return n


def left_rectangle(f, a, b, h):
    n = count_intervals(a, b, h)
    return h * math.fsum(f(a + i * h) for i in range(n))


def right_rectangle(f, a, b, h):
    n = count_intervals(a, b, h)
    return h * math.fsum(f(a + i * h) for i in range(1, n + 1))


def midpoint(f, a, b, h):
    n = count_intervals(a, b, h)
    return h * math.fsum(f(a + (i + 0.5) * h) for i in range(n))


def trapezoid(f, a, b, h):
    n = count_intervals(a, b, h)
    return h * (0.5 * (f(a) + f(b)) + math.fsum(f(a + i*h) for i in range(1, n)))


def simpson(f, a, b, h):
    n = count_intervals(a, b, h)
    if n % 2:
        raise ValueError('Для метода Симпсона число интервалов должно быть чётным.')
    return h / 3 * (f(a) + f(b) + math.fsum((4 if i % 2 else 2) * f(a + i*h) for i in range(1, n)))


def runge_romberg(coarse, fine, h1, h2, p):
    correction = (fine - coarse) / ((h1 / h2)**p - 1)
    return fine + correction, abs(correction)


def main():
    if not H1 > H2 > 0:
        raise ValueError('Необходимо H1 > H2 > 0.')

    methods = [
        ('Левые прямоугольники', left_rectangle, 1),
        ('Правые прямоугольники', right_rectangle, 1),
        ('Средние прямоугольники', midpoint, 2),
        ('Трапеции', trapezoid, 2),
        ('Симпсон', simpson, 4)
    ]
    rows = []

    for name, method, p in methods:
        coarse, fine = method(f, A, B, H1), method(f, A, B, H2)
        refined, estimate = runge_romberg(coarse, fine, H1, H2, p)
        rows.append((name, coarse, fine, p, refined, estimate))

    print(f'Пределы: [{A:g}, {B:g}]; h1 = {H1:g}, h2 = {H2:g}')
    print(f'{"Метод":<24} {"I(h1)":>14} {"I(h2)":>14} {"p":>3} {"I Р–Р":>14} {"Оценка h2":>14}')

    for name, coarse, fine, p, refined, estimate in rows:
        print(f'{name:<24} {coarse:14.9f} {fine:14.9f} {p:3d} {refined:14.9f} {estimate:14.9f}')

    if EXACT is not None:
        print(f'\nТочное значение: {EXACT:.12f}')
        print(f'{"Метод":<24} {"|I(h2)-I|":>16} {"|I Р–Р - I|":>16}')

        for name, _, fine, _, refined, _ in rows:
            print(f'{name:<24} {abs(fine-EXACT):16.12f} {abs(refined-EXACT):16.12f}')

        best_fine = min(rows, key=lambda row: abs(row[2] - EXACT))
        best_refined = min(rows, key=lambda row: abs(row[4] - EXACT))

        print(f'\nНаиболее точный при h2: {best_fine[0]}.')
        print(f'Наиболее точный после уточнения: {best_refined[0]}.')

if __name__ == '__main__':
    main()
