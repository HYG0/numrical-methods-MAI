X = (0.0, 0.5, 1.0, 1.5, 2.0)
Y = (1.0, 1.3776, 1.5403, 1.5707, 1.5839)
X_STAR = 1.0


def derivatives(x, y, x_star):
    if len(x) != len(y) or len(x) < 3:
        raise ValueError('Нужны не менее трёх пар значений x и y.')
    if any(x[i] >= x[i + 1] for i in range(len(x) - 1)):
        raise ValueError('Узлы x должны строго возрастать.')
    if not x[0] <= x_star <= x[-1]:
        raise ValueError('Точка вычисления должна находиться внутри таблицы.')

    interval = next((i for i in range(len(x) - 1) if x[i] <= x_star < x[i + 1]), len(x) - 2)
    start = max(0, min(interval - 1, len(x) - 3))

    x0, x1, x2 = x[start:start + 3]
    y0, y1, y2 = y[start:start + 3]

    slope01 = (y1 - y0) / (x1 - x0)
    slope12 = (y2 - y1) / (x2 - x1)

    divided_difference = (slope12 - slope01) / (x2 - x0)
    first = slope01 + divided_difference * (2 * x_star - x0 - x1)
    second = 2 * divided_difference

    return first, second, tuple(zip(x[start:start + 3], y[start:start + 3]))


def main():
    print('Исходная таблица:')
    print(f'{"i":>3} {"x_i":>8} {"y_i":>10}')

    for i, (x, y) in enumerate(zip(X, Y)):
        print(f'{i:3d} {x:8.1f} {y:10.4f}')

    first, second, nodes = derivatives(X, Y, X_STAR)

    print(f'\nТочка вычисления: x* = {X_STAR:.1f}')

    print('Использованные узлы (x_i, y_i):')
    for x, y in nodes:
        print(f'  ({x:.1f}, {y:.4f})')

    print(f"Первая производная:  f'(x*)  ≈ {first:.6f}")
    print(f"Вторая производная: f''(x*) ≈ {second:.6f}")


if __name__ == '__main__':
    main()
