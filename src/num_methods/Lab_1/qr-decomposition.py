from numpy._core.numeric import sign
import numpy as np
from os import path

DATA_TYPE = np.float64
FILES_PATH = path.dirname(path.abspath(__file__)) + '/'
DEBUG_FILES_PATH = path.abspath('.') + '/src/num_methods/Lab_1/'

np.set_printoptions(precision=3)

EPS = 1e-2

# Параметры QR-алгоритма поиска собственных значений
MAX_ITER_QR = 100_000
EPS_QR = 1e-10


def create_matrix(n : int):
    matrix = np.array([ [float(x) for x in input().split()] for _ in range(n) ], dtype=DATA_TYPE)
    return matrix

def read_matrix_from_file(file_name):
    with open(FILES_PATH + file_name) as f:
        rows = f.readlines()
        matrix = np.array([ [float(x) for x in row.split()] for row in rows ], dtype=DATA_TYPE)

    return matrix

def qr_decomposition(A: np.ndarray):
    """
    QR-decomposition for matrix A (NxN)

    return:
        Q, R
    """
    size = A.shape[0]
    A_k = A.copy()
    Q = np.eye(size)

    for i in range(size - 1):  # n - 1 итерация
        # Высчитываем вектор-строку (-столбец) для обнуления
        # поддиагональных элементов i-го столбца матрицы A_k
        v = np.zeros((1, size))

        v_ii = A_k[i, i] + sign(A_k[i, i]) * np.linalg.norm(A_k[:, i], ord=2)

        v[0, i] = v_ii
        # копируем поддиагональные элементы i-го столбца матрицы A_k
        v[0, i + 1:] = A_k[i + 1:, i]

        v = v.reshape(-1, 1)  # превращаем в вектор-столбец
        E = np.eye(size)

        # Строим матрицу Хаусхолдера H
        H = E - 2  / np.dot(v.T, v) * np.dot(v, v.T)

        # R <=> An-1
        # Ak = Hk * Ak-1
        A_k = np.dot(H, A_k)

        # Q = H1 * H2 * ... * Hn-1
        Q = np.dot(Q, H)

    return Q, A_k


def _extract_2x2_eigenvalues(block) -> tuple:
    """Собственные значения блока [[a, b], [c, d]].

    lambda1,2 = (a+d)/2 ± sqrt(((a-d)/2)^2 + b*c)
    Корень берётся через np.sqrt(complex(D)): при D < 0 получается комплексная пара.
    """
    a, b = complex(block[0, 0]), complex(block[0, 1])
    c, d = complex(block[1, 0]), complex(block[1, 1])

    m = (a + d) / 2
    D = ((a - d) / 2) ** 2 + b * c
    s = np.sqrt(complex(D))
    return (m + s, m - s)


def _wilkinson_shift(A) -> complex | float:
    """Сдвиг Уилкинсона: корень нижнего 2×2 блока, ближайший к A[-1,-1].

    Если выбранный корень комплексный (|imag| > EPS_QR), возвращается 0.0
    (без сдвига): комплексная пара будет извлечена из изолированного 2×2 блока
    при дефляции.
    """
    n = A.shape[0]
    if n == 1:
        return A[0, 0]

    l1, l2 = _extract_2x2_eigenvalues(A[-2:, -2:])
    target = A[-1, -1]
    mu = l1 if abs(l1 - target) <= abs(l2 - target) else l2

    if abs(np.imag(mu)) > EPS_QR:
        return 0.0
    return float(np.real(mu))


def _row_is_deflatable(A, i: int, tol: float) -> bool:
    """Строку i можно дефлировать: все элементы левее диагонали ~ 0.

    Тогда A = [[A', x], [0, lambda]] и A[i, i] — собственное значение.
    """
    return bool(np.all(np.abs(A[i, :i]) < tol))


def _block_is_isolated(A, q: int, tol: float) -> bool:
    """2×2 блок (строки/столбцы q-1, q) изолирован: столбцы 0..q-2 в этих строках ~ 0.

    Тогда A = [[A', X], [0, B]] и корни блока B входят в спектр.
    """
    return bool(np.all(np.abs(A[q - 1:q + 1, :q - 1]) < tol))


def QR_eigenvalues(A: np.ndarray, max_iter: int = MAX_ITER_QR, eps: float = EPS_QR) -> np.ndarray:
    """Собственные значения квадратной матрицы QR-алгоритмом со сдвигом Уилкинсона.

    Возвращает np.ndarray длины n (элементы могут быть комплексными).
    Порядок значений произвольный — сравнивать как мультимножество.
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Ожидалась квадратная матрица, получен размер {A.shape}")

    n = A.shape[0]
    if n == 0:
        return np.array([], dtype=DATA_TYPE)

    A_k = A.copy().astype(DATA_TYPE)
    if n == 1:
        return np.array([A_k[0, 0]], dtype=DATA_TYPE)

    # Порог дефляции, масштабированный по норме матрицы
    tol = eps * max(1.0, np.linalg.norm(A_k, ord=np.inf))
    eigenvalues = []
    p, q = 0, n - 1  # активное окно [p, q]

    while q >= p:
        # 1) Дефляция снизу: вся строка q ниже диагонали ~ 0
        while q > p and _row_is_deflatable(A_k, q, tol):
            eigenvalues.append(A_k[q, q])
            q -= 1
        if q == p:
            eigenvalues.append(A_k[p, p])
            break

        # 2) Изолированный 2×2 блок снизу (или активное окно размером 2):
        #    оба корня извлекаются точно, итерации не нужны
        if q == p + 1 or (q > p + 1 and _block_is_isolated(A_k, q, tol)):
            block = A_k[q - 1:q + 1, q - 1:q + 1]
            eigenvalues.extend(_extract_2x2_eigenvalues(block))
            q -= 2
            continue

        # 3) QR-итерация со сдвигом Уилкинсона на активном окне
        active = A_k[p:q + 1, p:q + 1].copy()
        size = q - p + 1
        I = np.eye(size, dtype=DATA_TYPE)

        for _ in range(max_iter):
            mu = _wilkinson_shift(active)
            shifted = active - mu * I
            Q, R = qr_decomposition(shifted)
            active = R @ Q + mu * I
            A_k[p:q + 1, p:q + 1] = active
            # остановка: готова одиночная дефляция ИЛИ изоляция 2×2 блока
            if _row_is_deflatable(A_k, q, tol) or _block_is_isolated(A_k, q, tol):
                break
        else:
            raise RuntimeError(
                f"QR_eigenvalues: сходимость не достигнута за {max_iter} итераций "
                f"(окно [{p}, {q}], max|A[q, :q]| = "
                f"{np.max(np.abs(A_k[q, :q])):.3e}, порог {tol:.3e})"
            )
        # возврат к началу цикла — дефляция/извлечение заберут сойденную часть

    return np.array(eigenvalues)


if __name__ == '__main__':
    matrix = read_matrix_from_file('qr_test')
    Q, R = qr_decomposition(matrix)

    print(f"===== Matrix A =====")
    print(matrix)
    print()

    print(f"===== Matrix Q =====")
    print(Q)
    print()

    print(f"===== Matrix R =====")
    print(R)
    print()

    print(f"===== Matrix Q·R =====")
    print(np.dot(Q, R))
    print()

    print(f"===== Eigenvalues matrix A =====")
    print(QR_eigenvalues(matrix))
