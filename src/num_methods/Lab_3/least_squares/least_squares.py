from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

X = np.array([-5.0, -3.0, -1.0, 1.0, 3.0, 5.0])
Y = np.array([2.9442, 2.8198, 2.3562, 0.7854, 0.32175, 0.1974])


def polynomial_text(coefficients):
    """Коэффициенты заданы в порядке от свободного члена к старшему."""
    terms = [f"{coefficients[0]:.8f}"]
    for power, coefficient in enumerate(coefficients[1:], 1):
        factor = "x" if power == 1 else f"x^{power}"
        terms.append(
            f" {'+' if coefficient >= 0 else '-'} {abs(coefficient):.8f}*{factor}"
        )
    return "".join(terms)


def main():
    output_dir = Path(__file__).resolve().parent
    models = {}
    for degree in (1, 2, 3):
        matrix = np.vander(X, N=degree + 1, increasing=True)
        coefficients, _, _, _ = np.linalg.lstsq(matrix, Y, rcond=None)
        mse = np.mean((matrix @ coefficients - Y) ** 2)
        models[degree] = (coefficients, mse)

        print(f"Степень {degree}:")

        for i, value in enumerate(coefficients):
            print(f"  a{i} = {value:.10f}")

        print(f"  P{degree}(x) = {polynomial_text(coefficients)}")
        print(f"  MSE{degree} = {mse:.10f}\n")

    print(
        "Сравнение ошибок: "
        + ", ".join(f"MSE{degree} = {mse:.10f}" for degree, (_, mse) in models.items())
    )
    best = min(models, key=lambda degree: models[degree][1])
    print(f"Наименьшая ошибка на исходных точках: P{best}(x).")

    grid = np.linspace(X.min(), X.max(), 400)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(X, Y, color="black", label="Исходные точки", zorder=5)

    for degree, (coefficients, mse) in models.items():
        values = np.polynomial.polynomial.polyval(grid, coefficients)
        ax.plot(grid, values, label=f"P{degree}(x), MSE = {mse:.6f}")

    ax.set(xlabel="x", ylabel="y", title="Метод наименьших квадратов")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = output_dir / "least_squares.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
