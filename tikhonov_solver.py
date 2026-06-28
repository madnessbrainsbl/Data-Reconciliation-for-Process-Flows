import logging
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize


MIN_VARIANCE = 1e-12
MIN_SIGMA = 1e-6
MIN_OBJECTIVE_SPAN = 1e-12
DEFAULT_INNER_MAXITER = 1000
DEFAULT_OUTER_MAXITER = 80
DEFAULT_LAMBDA_BOUNDS = (1e-8, 1e8)
DEFAULT_PILOT_POINTS = 60
FAILED_SOLVE_PENALTY = 1e30


@dataclass(frozen=True)
class RegularizedBalanceResult:
    """Result of one inner Tikhonov balance reconciliation solve."""

    x: np.ndarray
    lambda_reg: float
    balance_term: float
    delta_term: float
    objective: float
    success: bool
    message: str
    nit: int


@dataclass(frozen=True)
class LambdaOptimizationResult:
    """Result of lambda selection around inner SLSQP solves."""

    best_lambda: float
    best_result: RegularizedBalanceResult
    outer_success: bool
    outer_message: str
    pilot: dict[str, np.ndarray]
    outer_trace: dict[str, np.ndarray]
    additive_best_lambda: float
    multiplicative_best_lambda: float
    lcurve_best_lambda: float


def _as_float_vector(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} contains non-finite values")
    return vector


def _as_float_matrix(values: Sequence[Sequence[float]] | np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values")
    return matrix


def _validate_balance_inputs(
    A: Sequence[Sequence[float]] | np.ndarray,
    x0: Sequence[float] | np.ndarray,
    std_devs: Sequence[float] | np.ndarray,
    balance_targets: Sequence[float] | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    incidence = _as_float_matrix(A, "A")
    measurements = _as_float_vector(x0, "x0")
    sigmas = np.maximum(_as_float_vector(std_devs, "std_devs"), MIN_SIGMA)

    if incidence.shape[1] != measurements.size:
        raise ValueError("A column count must match x0 length")
    if measurements.size != sigmas.size:
        raise ValueError("x0 length must match std_devs length")

    if balance_targets is None:
        targets = np.zeros(incidence.shape[0], dtype=float)
    else:
        targets = _as_float_vector(balance_targets, "balance_targets")
        if targets.size != incidence.shape[0]:
            raise ValueError("balance_targets length must match A row count")

    return incidence, measurements, sigmas, targets


def _normalize_bounds(
    bounds: Sequence[tuple[float | None, float | None]] | None,
    n_values: int,
) -> list[tuple[float | None, float | None]]:
    if bounds is None:
        return [(0.0, None) for _ in range(n_values)]
    normalized = list(bounds)
    if len(normalized) != n_values:
        raise ValueError("bounds length must match x0 length")
    return normalized


def _clip_to_bounds(
    values: np.ndarray,
    bounds: Sequence[tuple[float | None, float | None]],
) -> np.ndarray:
    clipped = values.astype(float, copy=True)
    for index, (lower, upper) in enumerate(bounds):
        if lower is not None:
            clipped[index] = max(clipped[index], lower)
        if upper is not None:
            clipped[index] = min(clipped[index], upper)
    return clipped


def _node_sigmas(A: np.ndarray, std_devs: np.ndarray) -> np.ndarray:
    node_variances = (A**2).dot(np.maximum(std_devs**2, MIN_VARIANCE))
    return np.sqrt(np.maximum(node_variances, MIN_VARIANCE))


def _score_span(values: np.ndarray) -> float:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return 1.0
    return max(float(np.max(finite_values) - np.min(finite_values)), MIN_OBJECTIVE_SPAN)


def _normalize_scores(values: np.ndarray) -> np.ndarray:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return np.ones_like(values, dtype=float)
    minimum = float(np.min(finite_values))
    span = _score_span(finite_values)
    normalized = (values - minimum) / span
    return np.where(np.isfinite(normalized), normalized, FAILED_SOLVE_PENALTY)


def _lcurve_norms(balance_terms: np.ndarray, delta_terms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    residual_norms = np.sqrt(np.maximum(balance_terms, MIN_OBJECTIVE_SPAN))
    stabilizer_norms = np.sqrt(np.maximum(delta_terms, MIN_OBJECTIVE_SPAN))
    return residual_norms, stabilizer_norms


def _lcurve_curvature(balance_terms: np.ndarray, delta_terms: np.ndarray) -> np.ndarray:
    residual_norms, stabilizer_norms = _lcurve_norms(balance_terms, delta_terms)
    finite_mask = (
        np.isfinite(residual_norms)
        & np.isfinite(stabilizer_norms)
        & (balance_terms > MIN_OBJECTIVE_SPAN)
        & (delta_terms > MIN_OBJECTIVE_SPAN)
    )
    finite_indices = np.flatnonzero(finite_mask)
    curvature = np.zeros_like(residual_norms, dtype=float)

    if finite_indices.size < 3:
        return curvature

    log_residual = np.log10(residual_norms[finite_indices])
    log_stabilizer = np.log10(stabilizer_norms[finite_indices])
    d_residual = np.gradient(log_residual)
    d_stabilizer = np.gradient(log_stabilizer)
    dd_residual = np.gradient(d_residual)
    dd_stabilizer = np.gradient(d_stabilizer)
    denominator = np.maximum((d_residual**2 + d_stabilizer**2) ** 1.5, MIN_OBJECTIVE_SPAN)
    finite_curvature = np.abs(d_residual * dd_stabilizer - d_stabilizer * dd_residual) / denominator
    finite_curvature[0] = 0.0
    finite_curvature[-1] = 0.0
    curvature[finite_indices] = finite_curvature
    return curvature


def _lcurve_corner_index(balance_terms: np.ndarray, delta_terms: np.ndarray) -> int:
    curvature = _lcurve_curvature(balance_terms, delta_terms)
    if np.any(curvature > 0.0):
        return int(np.argmax(curvature))

    finite_mask = np.isfinite(balance_terms) & np.isfinite(delta_terms)
    finite_indices = np.flatnonzero(finite_mask)
    if finite_indices.size == 0:
        return 0
    if finite_indices.size < 3:
        finite_scores = balance_terms[finite_indices] + delta_terms[finite_indices]
        return int(finite_indices[int(np.argmin(finite_scores))])
    return int(finite_indices[finite_indices.size // 2])


def solve_regularized_balance(
    A: Sequence[Sequence[float]] | np.ndarray,
    x0: Sequence[float] | np.ndarray,
    std_devs: Sequence[float] | np.ndarray,
    lambda_reg: float,
    bounds: Sequence[tuple[float | None, float | None]] | None = None,
    balance_targets: Sequence[float] | np.ndarray | None = None,
    maxiter: int = DEFAULT_INNER_MAXITER,
    ftol: float = 1e-10,
) -> RegularizedBalanceResult:
    """
    Inner problem from the paper:
        min_x ||A x - b||^2_sigma_node + lambda * ||x - x0||^2_sigma

    Bounds are physical limits, not confidence corridors. For plant flows the
    default is x >= 0.
    """
    if not np.isfinite(lambda_reg) or lambda_reg < 0.0:
        raise ValueError("lambda_reg must be a finite non-negative value")

    incidence, measurements, sigmas, targets = _validate_balance_inputs(
        A,
        x0,
        std_devs,
        balance_targets,
    )
    normalized_bounds = _normalize_bounds(bounds, measurements.size)
    start = _clip_to_bounds(measurements, normalized_bounds)
    node_sigmas = _node_sigmas(incidence, sigmas)
    stream_variances = np.maximum(sigmas**2, MIN_VARIANCE)
    node_variances = np.maximum(node_sigmas**2, MIN_VARIANCE)

    def objective(values: np.ndarray) -> float:
        balance_residual = incidence.dot(values) - targets
        normalized_balance = balance_residual / node_sigmas
        normalized_delta = (values - measurements) / sigmas
        balance_term = float(np.dot(normalized_balance, normalized_balance))
        delta_term = float(np.dot(normalized_delta, normalized_delta))
        return balance_term + lambda_reg * delta_term

    def gradient(values: np.ndarray) -> np.ndarray:
        balance_residual = incidence.dot(values) - targets
        balance_grad = 2.0 * incidence.T.dot(balance_residual / node_variances)
        delta_grad = 2.0 * lambda_reg * (values - measurements) / stream_variances
        return balance_grad + delta_grad

    scipy_result = minimize(
        objective,
        start,
        jac=gradient,
        method="SLSQP",
        bounds=normalized_bounds,
        options={"maxiter": maxiter, "ftol": ftol, "disp": False},
    )

    x_rec = np.asarray(scipy_result.x, dtype=float)
    balance_residual = incidence.dot(x_rec) - targets
    normalized_balance = balance_residual / node_sigmas
    normalized_delta = (x_rec - measurements) / sigmas
    balance_term = float(np.dot(normalized_balance, normalized_balance))
    delta_term = float(np.dot(normalized_delta, normalized_delta))
    objective_value = balance_term + lambda_reg * delta_term

    return RegularizedBalanceResult(
        x=x_rec,
        lambda_reg=float(lambda_reg),
        balance_term=balance_term,
        delta_term=delta_term,
        objective=float(objective_value),
        success=bool(scipy_result.success),
        message=str(scipy_result.message),
        nit=int(getattr(scipy_result, "nit", 0)),
    )


def _build_lambda_diagnostics(
    lambdas: np.ndarray,
    solver: Callable[[float], RegularizedBalanceResult],
) -> dict[str, np.ndarray]:
    balance_terms: list[float] = []
    delta_terms: list[float] = []
    objectives: list[float] = []
    successes: list[float] = []

    for lambda_value in lambdas:
        try:
            result = solver(float(lambda_value))
        except (ValueError, FloatingPointError):
            balance_terms.append(np.inf)
            delta_terms.append(np.inf)
            objectives.append(np.inf)
            successes.append(0.0)
            continue

        balance_terms.append(result.balance_term)
        delta_terms.append(result.delta_term)
        objectives.append(result.objective)
        successes.append(1.0 if result.success else 0.0)

    balance_array = np.asarray(balance_terms, dtype=float)
    delta_array = np.asarray(delta_terms, dtype=float)
    residual_norms, stabilizer_norms = _lcurve_norms(balance_array, delta_array)
    lcurve_curvatures = _lcurve_curvature(balance_array, delta_array)
    normalized_balance = _normalize_scores(balance_array)
    normalized_delta = _normalize_scores(delta_array)
    additive_scores = normalized_balance + normalized_delta
    multiplicative_scores = (normalized_balance + MIN_OBJECTIVE_SPAN) * (
        normalized_delta + MIN_OBJECTIVE_SPAN
    )
    lcurve_idx = _lcurve_corner_index(balance_array, delta_array)

    return {
        "lambdas": lambdas,
        "log_lambdas": np.log10(lambdas),
        "balance_terms": balance_array,
        "delta_terms": delta_array,
        "lcurve_residual_norms": residual_norms,
        "lcurve_stabilizer_norms": stabilizer_norms,
        "lcurve_curvatures": lcurve_curvatures,
        "objectives": np.asarray(objectives, dtype=float),
        "successes": np.asarray(successes, dtype=float),
        "normalized_balance": normalized_balance,
        "normalized_delta": normalized_delta,
        "additive_scores": additive_scores,
        "multiplicative_scores": multiplicative_scores,
        "lcurve_idx": np.asarray([lcurve_idx], dtype=int),
    }


def optimize_lambda_nested(
    A: Sequence[Sequence[float]] | np.ndarray,
    x0: Sequence[float] | np.ndarray,
    std_devs: Sequence[float] | np.ndarray,
    lambda_bounds: tuple[float, float] = DEFAULT_LAMBDA_BOUNDS,
    bounds: Sequence[tuple[float | None, float | None]] | None = None,
    balance_targets: Sequence[float] | np.ndarray | None = None,
    pilot_points: int = DEFAULT_PILOT_POINTS,
    maxiter_outer: int = DEFAULT_OUTER_MAXITER,
) -> LambdaOptimizationResult:
    """
    Choose lambda by the L-curve method: maximum curvature in coordinates
    log(||A x_lambda - b||) and log(||x_lambda - x0||).
    """
    lower, upper = lambda_bounds
    if lower <= 0.0 or upper <= lower:
        raise ValueError("lambda_bounds must be positive and increasing")
    if pilot_points < 3:
        raise ValueError("pilot_points must be at least 3")

    incidence, measurements, sigmas, targets = _validate_balance_inputs(
        A,
        x0,
        std_devs,
        balance_targets,
    )
    normalized_bounds = _normalize_bounds(bounds, measurements.size)

    def solve(lambda_value: float) -> RegularizedBalanceResult:
        return solve_regularized_balance(
            incidence,
            measurements,
            sigmas,
            lambda_value,
            bounds=normalized_bounds,
            balance_targets=targets,
        )

    pilot_lambdas = np.logspace(np.log10(lower), np.log10(upper), pilot_points)
    pilot = _build_lambda_diagnostics(pilot_lambdas, solve)
    additive_index = int(np.nanargmin(pilot["additive_scores"]))
    multiplicative_index = int(np.nanargmin(pilot["multiplicative_scores"]))
    lcurve_index = int(pilot["lcurve_idx"][0])
    best_lambda = float(pilot_lambdas[lcurve_index])
    best_result = solve(best_lambda)

    if not best_result.success:
        finite_indices = np.flatnonzero(np.isfinite(pilot["balance_terms"]) & np.isfinite(pilot["delta_terms"]))
        if finite_indices.size:
            nearest_index = int(finite_indices[np.argmin(np.abs(finite_indices - lcurve_index))])
            best_lambda = float(pilot_lambdas[nearest_index])
            best_result = solve(best_lambda)
        if not best_result.success:
            fallback_index = int(np.nanargmin(pilot["additive_scores"]))
            best_lambda = float(pilot_lambdas[fallback_index])
            best_result = solve(best_lambda)

    outer_trace = {
        "log_lambdas": pilot["log_lambdas"],
        "lambdas": pilot["lambdas"],
        "scores": pilot["lcurve_curvatures"],
        "balance_terms": pilot["balance_terms"],
        "delta_terms": pilot["delta_terms"],
    }

    return LambdaOptimizationResult(
        best_lambda=best_lambda,
        best_result=best_result,
        outer_success=bool(best_result.success),
        outer_message="L-curve corner selected by maximum curvature",
        pilot=pilot,
        outer_trace=outer_trace,
        additive_best_lambda=float(pilot_lambdas[additive_index]),
        multiplicative_best_lambda=float(pilot_lambdas[multiplicative_index]),
        lcurve_best_lambda=float(pilot_lambdas[lcurve_index]),
    )


def auto_select_lambda(A, x_meas, std_devs, method="lcurve",
                       lambda_range=None, n_points=50):
    """
    Legacy-подбор параметра регуляризации λ.

    Не использовать для схемы цеха и методики статьи. Для нее нужен
    optimize_lambda_nested(), где решается внутренняя SLSQP-задача
    min ||(A x - b)/σузл||² + λ ||(x - x0)/σ||².

    Методы:
      - 'lcurve'  : L-кривая — максимизация кривизны в пространстве
                     (log ||x - x_meas||²_W, log λ||x||²)
      - 'gcv'     : Обобщённая перекрёстная проверка (Generalized Cross-Validation)

    Возвращает:
      best_lambda, diagnostics_dict
    """
    if lambda_range is None:
        lambda_range = np.logspace(-6, 2, n_points)

    variances = np.maximum(std_devs ** 2, 1e-10)
    W_diag = 1.0 / variances
    n = len(x_meas)

    residual_norms = []
    reg_norms = []
    gcv_scores = []
    lambdas_tested = []

    for lam in lambda_range:
        rec = TikhonovReconciler(A, lambda_reg=lam)
        x_rec, _, _, _ = rec.reconcile(x_meas, std_devs)

        r_norm = np.sum(W_diag * (x_rec - x_meas) ** 2)
        x_norm = np.sum(x_rec ** 2)
        residual_norms.append(r_norm)
        reg_norms.append(lam * x_norm)

        # GCV: score = (1/n) * ||W^{1/2}(x-y)||^2 / (1 - tr(H)/n)^2
        # Упрощённая оценка: H ≈ W(W + λI)^{-1}
        # tr(H) ≈ Σ w_i / (w_i + λ)
        tr_H = np.sum(W_diag / (W_diag + lam))
        denom = (1.0 - tr_H / n) ** 2
        if denom > 1e-12:
            gcv = r_norm / n / denom
        else:
            gcv = np.inf
        gcv_scores.append(gcv)
        lambdas_tested.append(lam)

    residual_norms = np.array(residual_norms)
    reg_norms = np.array(reg_norms)
    gcv_scores = np.array(gcv_scores)
    lambdas_tested = np.array(lambdas_tested)

    if method == "gcv":
        best_idx = np.argmin(gcv_scores)
    else:
        # L-кривая: максимизация кривизны
        log_r = np.log10(residual_norms + 1e-30)
        log_x = np.log10(reg_norms + 1e-30)
        # Кривизна через конечные разности
        dr = np.gradient(log_r)
        dx = np.gradient(log_x)
        ddr = np.gradient(dr)
        ddx = np.gradient(dx)
        curvature = np.abs(dr * ddx - dx * ddr) / (dr**2 + dx**2)**1.5
        curvature[0] = 0
        curvature[-1] = 0
        best_idx = np.argmax(curvature)

    best_lambda = lambdas_tested[best_idx]

    diagnostics = {
        "lambdas": lambdas_tested,
        "residual_norms": residual_norms,
        "reg_norms": reg_norms,
        "gcv_scores": gcv_scores,
        "best_idx": best_idx,
        "best_lambda": best_lambda,
        "method": method,
    }

    return best_lambda, diagnostics

class TikhonovReconciler:
    """
    Решатель задачи сведения материального баланса
    методом взвешенных наименьших квадратов (WLS)
    с РЕЛАКСАЦИЕЙ ограничений на измеренные значения.

    Постановка задачи:
      Минимизировать:  J(x) = Σ w_i (x_i - x_meas_i)²
      При условии:     A x = 0                          (строгий баланс)
      При условии:     x_meas_i - σ_i ≤ x_i ≤ x_meas_i + σ_i   (релаксация)

    Где:
      - w_i = 1/σ_i²  — весовой коэффициент (обратная дисперсия)
      - σ_i            — погрешность прибора (из паспорта / DOCX)
      - A              — матрица инцидентности (топология сети)
    """
    def __init__(self, incidence_matrix, lambda_reg=0.0):
        self.A = np.array(incidence_matrix)
        self.n_nodes = self.A.shape[0]
        self.n_streams = self.A.shape[1]
        self.lambda_reg = lambda_reg
        self.logger = logging.getLogger("TikhonovReconciler")

    def reconcile(self, x_meas, std_devs):
        """
        Решение задачи SLSQP:
          min   J(x) = Σ w_i (x_i - x_meas_i)²  [+ λ·||x||² если lambda_reg > 0]
          s.t.  A·x = 0                           (строгий материальный баланс)
          s.t.  x_meas_i - σ_i ≤ x_i ≤ x_meas_i + σ_i  (box-ограничения / релаксация)
        """
        variances = np.maximum(std_devs ** 2, 1e-10)
        W_diag = 1.0 / variances

        # Целевая функция J(x)
        def objective(x):
            diff = x - x_meas
            J = np.sum(W_diag * diff**2)
            if self.lambda_reg > 0:
                J += self.lambda_reg * np.sum(x**2)
            return J

        # Градиент целевой функции (для ускорения сходимости SLSQP)
        def gradient(x):
            g = 2.0 * W_diag * (x - x_meas)
            if self.lambda_reg > 0:
                g += 2.0 * self.lambda_reg * x
            return g

        # Ограничение: A·x = 0 (строгий баланс, каждая строка — отдельное равенство)
        constraints = {
            'type': 'eq',
            'fun': lambda x: self.A.dot(x),
            'jac': lambda x: self.A,
        }

        # Box-ограничения (РЕЛАКСАЦИЯ):
        # x_meas_i - σ_i  ≤  x_i  ≤  x_meas_i + σ_i
        bounds = []
        for i in range(self.n_streams):
            lb = x_meas[i] - std_devs[i]
            ub = x_meas[i] + std_devs[i]
            bounds.append((lb, ub))

        # Начальное приближение = измерения
        x0 = x_meas.copy()

        # Запуск оптимизатора SLSQP
        result = minimize(
            objective, x0,
            jac=gradient,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-12, 'disp': False}
        )

        x_rec = result.x
        residuals = x_rec - x_meas
        balance_violation = self.A.dot(x_rec)

        # Расчёт компонент целевой функции
        J_wls = float(np.sum(W_diag * (residuals ** 2)))
        J_reg = float(self.lambda_reg * np.sum(x_rec ** 2)) if self.lambda_reg > 0 else 0.0
        J_total = J_wls + J_reg

        if not result.success:
            self.logger.warning(f"SLSQP не сошёлся: {result.message}")

        return x_rec, J_total, residuals, balance_violation

def print_tikhonov_analytics(stream_names, std_devs, x_meas, x_rec,
                             J_total, balance_violation=None, A=None):
    """
    Аналитическая таблица результатов сведения баланса.
    """
    print("\n" + "="*105)
    print(f"АНАЛИТИЧЕСКАЯ ТАБЛИЦА: Сведение балансов (WLS + Релаксация)")
    print(f"Общее значение целевой функции J = {J_total:.6f}")
    print("="*105)

    header = (f"{'Поток':<18} | {'σ':<8} | {'Измерено':<10} | {'Нижн.гр.':<10} | "
              f"{'Скоррект.':<10} | {'Верхн.гр.':<10} | {'Δ':<10} | {'В диап?':<7}")
    print(header)
    print("-" * 105)

    for i in range(len(stream_names)):
        name = str(stream_names[i])[:17]
        sigma = std_devs[i]
        meas = x_meas[i]
        rec = x_rec[i]
        delta = rec - meas
        lb = meas - sigma
        ub = meas + sigma
        in_range = "  Да" if lb - 1e-6 <= rec <= ub + 1e-6 else " НЕТ!"

        print(f"{name:<18} | {sigma:<8.3f} | {meas:<10.2f} | {lb:<10.2f} | "
              f"{rec:<10.2f} | {ub:<10.2f} | {delta:<10.4f} | {in_range:<7}")

    print("="*105)

    if balance_violation is not None:
        print(f"\nПРОВЕРКА БАЛАНСА (Ax = 0, строгое):")
        print("-" * 50)
        for j, bv in enumerate(balance_violation):
            status = "OK" if abs(bv) < 1e-4 else "НАРУШЕН"
            print(f"  Узел {j+1:<3}: Ax = {bv:>10.6f}   [{status}]")
        print(f"  Макс |Ax| = {np.max(np.abs(balance_violation)):.8f}")
    print("="*105)
