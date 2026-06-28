"""
Сведение материального баланса цеха регуляризацией Тихонова.

Модель следует постановке из статьи:
    min_x ||A x||^2_sigma_node + lambda * ||x - x0||^2_sigma

lambda выбирается методом L-кривой по максимуму кривизны.
"""

from __future__ import annotations

import csv
import logging
import os
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from sicc_lib import SICCSolver
from tikhonov_solver import (
    LambdaOptimizationResult,
    RegularizedBalanceResult,
    optimize_lambda_nested,
)


RESULTS_DIR = Path("plant_results_lcurve_lambda")
PLOT_DPI = 220
FONT_FAMILY = "Arial"
DEFAULT_MONTE_CARLO_TRIALS = 1000
MAIN_PILOT_POINTS = 60
ANALYTICS_NAME_WRAP_WIDTH = 18
STREAM_NAMES_WRAP_WIDTH = 34
ANALYTICS_TABLE_BBOX = [0.01, 0.30, 0.98, 0.58]
ANALYTICS_COLUMN_WIDTHS_WITH_SICC = [0.045, 0.17, 0.10, 0.065, 0.06, 0.065, 0.08, 0.07, 0.08, 0.065, 0.08, 0.045, 0.045]
STREAM_NAMES_COLUMN_WIDTHS = [0.12, 0.62, 0.26]
MONTE_CARLO_PILOT_POINTS = 10
LAMBDA_BOUNDS = (1e-8, 1e8)
ANOMALY_RELATIVE_THRESHOLD = 0.10
IDENTIFICATION_Z_THRESHOLD = 2.0
ZERO_DIVISION_EPS = 1e-9
CURRENT_GROSS_ERRORS = {
    "P1": 0.4400,
    "Q1": -0.2688,
    "R1": 0.1000,
    "P6": 0.1050,
    "Q3": -0.3405,
}

plt.rcParams["font.family"] = FONT_FAMILY
logging.getLogger("sicc_lib").setLevel(logging.WARNING)


def wrap_table_text(value: object, width: int) -> str:
    return "\n".join(textwrap.wrap(str(value), width=width, break_long_words=False))


@dataclass(frozen=True)
class SICCResult:
    x: np.ndarray | None
    bias_map: dict[str, float]
    status: str


@dataclass(frozen=True)
class PlantRunResult:
    x_meas: np.ndarray
    nested: RegularizedBalanceResult
    lambda_result: LambdaOptimizationResult
    sicc: SICCResult


@dataclass(frozen=True)
class MonteCarloRow:
    scenario: str
    trial: int
    lambda_reg: float
    nested_rmse: float
    nested_relative_mae: float
    measurement_rmse: float
    measurement_relative_mae: float
    sicc_rmse: float
    sicc_relative_mae: float
    nested_detection_rate: float
    sicc_detection_rate: float


class NetworkAdapter:
    """Adapter for the existing SICC solver."""

    def __init__(
        self,
        A: np.ndarray,
        x_meas: np.ndarray,
        std_devs: np.ndarray,
        stream_ids: Sequence[str],
    ) -> None:
        self._A = A
        self._x_meas = x_meas
        self._Q = np.diag(std_devs**2)
        self.stream_order = list(stream_ids)
        self._graph = nx.Graph()

        n_nodes, n_streams = A.shape
        for stream_index in range(n_streams):
            source = None
            target = None
            for node_index in range(n_nodes):
                if A[node_index, stream_index] == -1.0:
                    source = node_index
                if A[node_index, stream_index] == 1.0:
                    target = node_index
            if source is not None and target is not None:
                self._graph.add_edge(source, target, stream=stream_ids[stream_index])

    def get_incidence_matrix(self) -> np.ndarray:
        return self._A

    def get_measurements_vector(self) -> np.ndarray:
        return self._x_meas

    def get_covariance_matrix(self) -> np.ndarray:
        return self._Q

    def find_loops(self, indices: Sequence[int]) -> bool:
        return False

    def find_undirected_cycle(self, target_stream_name: str) -> list[str]:
        for source, target, data in self._graph.edges(data=True):
            if data.get("stream") != target_stream_name:
                continue
            graph_without_target = self._graph.copy()
            graph_without_target.remove_edge(source, target)
            try:
                path = nx.shortest_path(graph_without_target, source, target)
            except nx.NetworkXNoPath:
                return [target_stream_name]

            cycle_streams = [target_stream_name]
            for path_index in range(len(path) - 1):
                edge_data = self._graph.get_edge_data(path[path_index], path[path_index + 1])
                if edge_data and "stream" in edge_data:
                    cycle_streams.append(str(edge_data["stream"]))
            return cycle_streams
        return [target_stream_name]


def define_plant_network() -> dict[str, object]:
    """Define the plant-shop graph from the workshop scheme."""
    nodes = [
        "ОНПЗ",
        "ЦПП",
        "ПАО_ОК",
        "Парк_ППФ",
        "Отд_Переработки",
        "Парк_ПФ",
        "Полимеризация",
        "Факел",
        "Выходы_перераб",
    ]

    streams = [
        {"id": "P1", "name": "ППФ с ОНПЗ", "from": 0, "to": 3, "true": 22.0, "sigma_pct": 0.25},
        {"id": "P2", "name": "Пропилен рецикл с ЦПП", "from": 1, "to": 3, "true": 0.8, "sigma_pct": 0.141},
        {"id": "P3", "name": "ППФ на переработку", "from": 3, "to": 4, "true": 21.5, "sigma_pct": 0.224},
        {"id": "P4", "name": "Слив ППФ/ПФ из в/ц", "from": 3, "to": 8, "true": 0.5, "sigma_pct": 50.0},
        {"id": "P5", "name": "Некондиц.продукт", "from": 4, "to": 3, "true": 1.2, "sigma_pct": 0.224},
        {"id": "P6", "name": "Сдувки у/в на факел (ППФ)", "from": 3, "to": 7, "true": 2.0, "sigma_pct": 0.75},
        {"id": "P7", "name": "Поток ППФ на всас Н402", "from": 3, "to": 3, "true": 3.0, "sigma_pct": 50.0},
        {"id": "Q1", "name": "Пропилен с переработки", "from": 4, "to": 5, "true": 15.0, "sigma_pct": 0.224},
        {"id": "Q2", "name": "Пропилен с ПАО ОК", "from": 2, "to": 5, "true": 5.0, "sigma_pct": 0.224},
        {"id": "Q3", "name": "Пропилен на полимеризацию", "from": 5, "to": 6, "true": 19.0, "sigma_pct": 0.224},
        {"id": "Q4", "name": "Сдувки у/в на факел (ПФ)", "from": 5, "to": 7, "true": 1.0, "sigma_pct": 0.75},
        {"id": "Q5", "name": "Поток ПФ на всас Н404", "from": 5, "to": 5, "true": 2.0, "sigma_pct": 50.0},
        {"id": "R1", "name": "Топливный газ (отдувки)", "from": 4, "to": 8, "true": 0.8, "sigma_pct": 1.0},
        {"id": "R2", "name": "Пропановая фракция", "from": 4, "to": 8, "true": 4.5, "sigma_pct": 0.224},
    ]

    n_nodes = len(nodes)
    n_streams = len(streams)
    incidence = np.zeros((n_nodes, n_streams), dtype=float)
    for stream_index, stream in enumerate(streams):
        source = int(stream["from"])
        target = int(stream["to"])
        if source == target:
            continue
        incidence[source, stream_index] = -1.0
        incidence[target, stream_index] = 1.0

    x_true = np.array([float(stream["true"]) for stream in streams], dtype=float)
    sigma_pct = np.array([float(stream["sigma_pct"]) for stream in streams], dtype=float)
    std_devs = np.maximum(x_true * sigma_pct / 100.0, 0.01)
    internal_nodes = [3, 4, 5]
    incidence_internal = incidence[internal_nodes, :]

    return {
        "nodes": nodes,
        "streams": streams,
        "A": incidence,
        "A_internal": incidence_internal,
        "internal_nodes": internal_nodes,
        "x_true": x_true,
        "std_devs": std_devs,
        "n_nodes": n_nodes,
        "n_streams": n_streams,
    }


def generate_measurements(
    net: Mapping[str, object],
    seed: int = 42,
    gross_errors: Mapping[str, float] | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x_true = np.asarray(net["x_true"], dtype=float)
    std_devs = np.asarray(net["std_devs"], dtype=float)
    x_meas = x_true + rng.normal(0.0, std_devs)

    if gross_errors:
        stream_index = stream_indices(net)
        for stream_id, magnitude in gross_errors.items():
            x_meas[stream_index[stream_id]] += magnitude

    return np.maximum(x_meas, 0.0)


def stream_ids(net: Mapping[str, object]) -> list[str]:
    return [str(stream["id"]) for stream in net["streams"]]  # type: ignore[index]


def stream_indices(net: Mapping[str, object]) -> dict[str, int]:
    return {stream_id: index for index, stream_id in enumerate(stream_ids(net))}


def physical_bounds(net: Mapping[str, object]) -> list[tuple[float, None]]:
    return [(0.0, None) for _ in stream_ids(net)]


def solve_nested_for_measurements(
    net: Mapping[str, object],
    x_meas: np.ndarray,
    pilot_points: int = MAIN_PILOT_POINTS,
) -> LambdaOptimizationResult:
    return optimize_lambda_nested(
        np.asarray(net["A_internal"], dtype=float),
        x_meas,
        np.asarray(net["std_devs"], dtype=float),
        lambda_bounds=LAMBDA_BOUNDS,
        bounds=physical_bounds(net),
        pilot_points=pilot_points,
    )


def solve_sicc(net: Mapping[str, object], x_meas: np.ndarray) -> SICCResult:
    try:
        adapter = NetworkAdapter(
            np.asarray(net["A_internal"], dtype=float),
            x_meas,
            np.asarray(net["std_devs"], dtype=float),
            stream_ids(net),
        )
        solver = SICCSolver(adapter)
        _, _, _, x_hat, bias_map, status = solver.solve_sicc()
    except np.linalg.LinAlgError as error:
        return SICCResult(x=None, bias_map={}, status=f"SICC linear algebra error: {error}")
    except ValueError as error:
        return SICCResult(x=None, bias_map={}, status=f"SICC value error: {error}")

    return SICCResult(
        x=np.asarray(x_hat, dtype=float),
        bias_map={str(key): float(value) for key, value in bias_map.items()},
        status=str(status),
    )


def run_single_case(
    net: Mapping[str, object],
    gross_errors: Mapping[str, float],
    seed: int,
    pilot_points: int = MAIN_PILOT_POINTS,
) -> PlantRunResult:
    x_meas = generate_measurements(net, seed=seed, gross_errors=gross_errors)
    lambda_result = solve_nested_for_measurements(net, x_meas, pilot_points=pilot_points)
    sicc_result = solve_sicc(net, x_meas)
    return PlantRunResult(
        x_meas=x_meas,
        nested=lambda_result.best_result,
        lambda_result=lambda_result,
        sicc=sicc_result,
    )


def root_mean_square_error(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - expected) ** 2)))


def relative_mae(actual: np.ndarray, expected: np.ndarray) -> float:
    denominator = np.maximum(np.abs(expected), ZERO_DIVISION_EPS)
    return float(np.mean(np.abs((actual - expected) / denominator)))


def normalized_delta(x_rec: np.ndarray, x_meas: np.ndarray, std_devs: np.ndarray) -> np.ndarray:
    return (x_rec - x_meas) / np.maximum(std_devs, ZERO_DIVISION_EPS)


def detect_streams_by_delta(
    x_rec: np.ndarray,
    x_meas: np.ndarray,
    std_devs: np.ndarray,
) -> set[str]:
    z_scores = np.abs(normalized_delta(x_rec, x_meas, std_devs))
    ids = stream_ids(define_plant_network())
    return {ids[index] for index, score in enumerate(z_scores) if score >= IDENTIFICATION_Z_THRESHOLD}


def true_positive_rate(detected: set[str], expected_errors: Mapping[str, float]) -> float:
    if not expected_errors:
        return 1.0 if not detected else 0.0
    expected = set(expected_errors.keys())
    return len(detected & expected) / len(expected)


def ensure_results_dir(output_dir: Path = RESULTS_DIR) -> Path:
    output_dir.mkdir(exist_ok=True)
    return output_dir


def plot_lambda_optimization(
    lambda_result: LambdaOptimizationResult,
    filename: Path,
) -> None:
    pilot = lambda_result.pilot
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Подбор λ методом L-кривой", fontsize=24, fontweight="bold")

    residual_norms = pilot["lcurve_residual_norms"]
    stabilizer_norms = pilot["lcurve_stabilizer_norms"]
    curvatures = pilot["lcurve_curvatures"]
    selected_residual_norm = float(np.sqrt(max(lambda_result.best_result.balance_term, ZERO_DIVISION_EPS)))
    selected_stabilizer_norm = float(np.sqrt(max(lambda_result.best_result.delta_term, ZERO_DIVISION_EPS)))
    selected_curvature = float(curvatures[int(pilot["lcurve_idx"][0])])
    log_residual_norms = np.log10(np.maximum(residual_norms, ZERO_DIVISION_EPS))
    log_stabilizer_norms = np.log10(np.maximum(stabilizer_norms, ZERO_DIVISION_EPS))
    lcurve_mask = (
        np.isfinite(log_residual_norms)
        & np.isfinite(log_stabilizer_norms)
        & (pilot["balance_terms"] > ZERO_DIVISION_EPS)
        & (pilot["delta_terms"] > ZERO_DIVISION_EPS)
    )

    ax_lcurve = axes[0, 0]
    ax_lcurve.plot(log_residual_norms[lcurve_mask], log_stabilizer_norms[lcurve_mask], color="#111111", lw=2.5, marker="o", ms=5)
    ax_lcurve.scatter(np.log10(max(selected_residual_norm, ZERO_DIVISION_EPS)), np.log10(max(selected_stabilizer_norm, ZERO_DIVISION_EPS)), s=260, color="#dc2626", marker="*", label=f"λ*={lambda_result.best_lambda:.2e}")
    ax_lcurve.set_xlabel("lg ||A xλ - b||2, норма невязки", fontsize=15)
    ax_lcurve.set_ylabel("lg ||Lxλ||2, норма стабилизатора", fontsize=15)
    ax_lcurve.set_title("1. L-кривая: выбираем угол", fontsize=17, fontweight="bold")
    ax_lcurve.grid(True, alpha=0.35)
    ax_lcurve.legend(fontsize=13)

    ax_curvature = axes[0, 1]
    ax_curvature.semilogx(pilot["lambdas"], curvatures, color="#005bbb", lw=3, marker="o", ms=5, label="Кривизна L-кривой")
    ax_curvature.axvline(lambda_result.best_lambda, color="#dc2626", lw=3, linestyle="--", label=f"λ*={lambda_result.best_lambda:.2e}")
    ax_curvature.set_xlabel("λ", fontsize=15)
    ax_curvature.set_ylabel("Кривизна", fontsize=15)
    ax_curvature.set_title("2. Берем λ в максимуме кривизны", fontsize=17, fontweight="bold")
    ax_curvature.grid(True, alpha=0.35)
    ax_curvature.legend(fontsize=13)

    ax_outer = axes[1, 0]
    ax_outer.semilogx(pilot["lambdas"], residual_norms, color="#2563eb", lw=3, marker="o", ms=5, label="Невязка баланса")
    ax_outer.semilogx(pilot["lambdas"], stabilizer_norms, color="#f97316", lw=3, marker="s", ms=5, label="Стабилизатор")
    ax_outer.axvline(lambda_result.best_lambda, color="#dc2626", lw=3, linestyle="--")
    ax_outer.set_xlabel("λ", fontsize=15)
    ax_outer.set_ylabel("Норма", fontsize=15)
    ax_outer.set_title("3. Что меняется при разных λ", fontsize=17, fontweight="bold")
    ax_outer.grid(True, alpha=0.35)
    ax_outer.legend(fontsize=13)

    ax_table = axes[1, 1]
    ax_table.axis("off")
    table_rows = [
        ["Выбранная λ", f"{lambda_result.best_lambda:.6e}"],
        ["Как выбрана", "угол L-кривой"],
        ["Критерий", "максимум кривизны"],
        ["Норма невязки", f"{selected_residual_norm:.4f}"],
        ["Норма стабилизатора", f"{selected_stabilizer_norm:.4f}"],
        ["Кривизна", f"{selected_curvature:.4f}"],
        ["Расчет сошелся", "да" if lambda_result.best_result.success else "нет"],
    ]
    table = ax_table.table(cellText=table_rows, colLabels=["Метрика", "Значение"], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.2, 2.0)
    for column in range(2):
        table[0, column].set_facecolor("#1f2937")
        table[0, column].set_text_props(color="white", fontweight="bold")
    ax_table.set_title("Итог подбора", fontsize=17, fontweight="bold")

    plt.tight_layout()
    plt.savefig(filename, dpi=PLOT_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_plant_graph_readable(
    net: Mapping[str, object],
    x_meas: np.ndarray,
    x_rec: np.ndarray,
    gross_errors: Mapping[str, float],
    filename: Path,
) -> None:
    nodes = list(net["nodes"])  # type: ignore[arg-type]
    streams = list(net["streams"])  # type: ignore[arg-type]
    x_true = np.asarray(net["x_true"], dtype=float)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(nodes)))

    source_nodes = {0, 1, 2}
    sink_nodes = {6, 7, 8}
    node_colors = ["#22c55e" if i in source_nodes else "#ef4444" if i in sink_nodes else "#0ea5e9" for i in range(len(nodes))]
    positions = {
        0: (-5.2, 2.2),
        1: (-5.2, 0.0),
        2: (-5.2, -3.0),
        3: (-1.2, 0.9),
        4: (2.4, 0.9),
        5: (2.4, -2.7),
        6: (6.6, -2.7),
        7: (6.6, 0.9),
        8: (6.6, 2.8),
    }
    label_offsets = {
        "P1": (-0.8, 0.35),
        "P2": (-0.6, 0.20),
        "P3": (0.0, 0.34),
        "P4": (0.15, 0.70),
        "P5": (0.0, -0.38),
        "P6": (0.2, -0.70),
        "P7": (0.0, 0.75),
        "Q1": (0.25, -0.25),
        "Q2": (-1.7, 0.15),
        "Q3": (0.0, -0.25),
        "Q4": (0.8, 0.15),
        "Q5": (0.0, 0.70),
        "R1": (0.65, 0.50),
        "R2": (0.45, 0.95),
    }
    edge_rads = {
        "P1": 0.12,
        "P2": 0.08,
        "P3": 0.02,
        "P4": -0.28,
        "P5": -0.18,
        "P6": -0.22,
        "Q1": 0.08,
        "Q2": 0.12,
        "Q3": -0.05,
        "Q4": 0.16,
        "R1": 0.23,
        "R2": 0.35,
    }

    fig, ax = plt.subplots(figsize=(24, 13))
    nx.draw_networkx_nodes(graph, positions, ax=ax, node_size=7200, node_color=node_colors, edgecolors="#111111", linewidths=3.2)
    node_labels = {
        0: "ОНПЗ",
        1: "ЦПП",
        2: "ПАО\nОК",
        3: "Парк\nППФ",
        4: "Перера-\nботка",
        5: "Парк\nПФ",
        6: "Полиме-\nризация",
        7: "Факел",
        8: "Выходы",
    }
    nx.draw_networkx_labels(graph, positions, node_labels, ax=ax, font_size=14, font_weight="bold")

    for stream_index, stream in enumerate(streams):
        source = int(stream["from"])
        target = int(stream["to"])
        stream_id = str(stream["id"])
        sigma_pct = float(stream["sigma_pct"])
        relative_deviation = abs(x_rec[stream_index] - x_true[stream_index]) / max(abs(x_true[stream_index]), ZERO_DIVISION_EPS)
        edge_color = "#dc2626" if stream_id in gross_errors or relative_deviation > ANOMALY_RELATIVE_THRESHOLD else "#f59e0b" if sigma_pct >= 50.0 else "#0f172a"
        width = 3.8 if stream_id in gross_errors else 2.8

        if source == target:
            x_pos, y_pos = positions[source]
            ax.annotate(
                "",
                xy=(x_pos + 0.55, y_pos + 0.30),
                xytext=(x_pos - 0.05, y_pos + 0.55),
                arrowprops=dict(arrowstyle="-|>", color=edge_color, lw=width, connectionstyle="arc3,rad=1.0", mutation_scale=24),
            )
        else:
            nx.draw_networkx_edges(
                graph,
                positions,
                edgelist=[(source, target)],
                ax=ax,
                edge_color=edge_color,
                width=width,
                arrows=True,
                arrowsize=25,
                connectionstyle=f"arc3,rad={edge_rads.get(stream_id, 0.05)}",
                min_source_margin=35,
                min_target_margin=35,
            )

        x1, y1 = positions[source]
        x2, y2 = positions[target]
        x_label = (x1 + x2) / 2.0
        y_label = (y1 + y2) / 2.0
        dx, dy = label_offsets.get(stream_id, (0.0, 0.0))
        ax.text(
            x_label + dx,
            y_label + dy,
            f"{stream_id}\n{x_meas[stream_index]:.2f} -> {x_rec[stream_index]:.2f}",
            fontsize=13,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=edge_color, linewidth=1.7, alpha=0.95),
            color="#111827",
            zorder=10,
        )

    legend_elements = [
        mpatches.Patch(facecolor="#22c55e", edgecolor="#111111", label="Источник"),
        mpatches.Patch(facecolor="#0ea5e9", edgecolor="#111111", label="Процесс"),
        mpatches.Patch(facecolor="#ef4444", edgecolor="#111111", label="Сток"),
        plt.Line2D([0], [0], color="#0f172a", lw=3, label="Основной поток"),
        plt.Line2D([0], [0], color="#f59e0b", lw=3, label="Без прибора учета"),
        plt.Line2D([0], [0], color="#dc2626", lw=4, label="Внесенная/крупная ошибка"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=15, framealpha=0.95)
    ax.set_title("Технологическая схема цеха: измерено => сведено регуляризацией Тихонова", fontsize=23, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=PLOT_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_corridor_readable(
    net: Mapping[str, object],
    x_meas: np.ndarray,
    x_rec: np.ndarray,
    x_sicc: np.ndarray | None,
    filename: Path,
) -> None:
    ids = stream_ids(net)
    x_true = np.asarray(net["x_true"], dtype=float)
    std_devs = np.asarray(net["std_devs"], dtype=float)
    indices = np.arange(len(ids))

    fig, ax = plt.subplots(figsize=(22, 9))
    ax.fill_between(indices, x_meas - 2.0 * std_devs, x_meas + 2.0 * std_devs, alpha=0.18, color="#fb7185", label="+/-2σ")
    ax.fill_between(indices, x_meas - std_devs, x_meas + std_devs, alpha=0.35, color="#fb7185", label="+/-1σ")
    ax.plot(indices, x_meas, "o-", color="#dc2626", ms=9, lw=2.8, label="Измерения")
    ax.plot(indices, x_true, "s--", color="#16a34a", ms=9, lw=2.8, label="Истина")
    ax.plot(indices, x_rec, "^-", color="#2563eb", ms=10, lw=3.2, label="Тихонов")
    if x_sicc is not None:
        ax.plot(indices, x_sicc, "D-", color="#9333ea", ms=9, lw=3.0, label="SICC")
    ax.set_xticks(indices)
    ax.set_xticklabels(ids, rotation=35, fontsize=14, fontweight="bold")
    ax.set_ylabel("Расход, т/ч", fontsize=17, fontweight="bold")
    ax.set_title("Коридор погрешностей и результаты сведения", fontsize=22, fontweight="bold")
    ax.legend(fontsize=15, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.45)
    ax.tick_params(axis="y", labelsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=PLOT_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_delta(
    net: Mapping[str, object],
    x_meas: np.ndarray,
    x_rec: np.ndarray,
    gross_errors: Mapping[str, float],
    filename: Path,
) -> None:
    ids = stream_ids(net)
    std_devs = np.asarray(net["std_devs"], dtype=float)
    delta = x_rec - x_meas
    z_delta = normalized_delta(x_rec, x_meas, std_devs)
    colors = ["#dc2626" if stream_id in gross_errors else "#2563eb" for stream_id in ids]
    indices = np.arange(len(ids))

    fig, axes = plt.subplots(2, 1, figsize=(21, 12), sharex=True)
    axes[0].bar(indices, delta, color=colors, alpha=0.9)
    axes[0].axhline(0.0, color="#111111", lw=2)
    axes[0].set_ylabel("δ = x - x0, т/ч", fontsize=16, fontweight="bold")
    axes[0].set_title("Абсолютная поправка потоков", fontsize=20, fontweight="bold")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    axes[0].text(
        0.985,
        0.92,
        "Красный столбец = поток с внесенной ошибкой.\nВысота столбца = насколько метод изменил измерение.",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=13,
        color="#7f1d1d",
        bbox=dict(facecolor="white", edgecolor="#dc2626", boxstyle="round,pad=0.45", alpha=0.94),
    )

    axes[1].bar(indices, z_delta, color=colors, alpha=0.9)
    for threshold in (-IDENTIFICATION_Z_THRESHOLD, IDENTIFICATION_Z_THRESHOLD):
        axes[1].axhline(threshold, color="#dc2626", lw=2.5, linestyle="--")
    axes[1].axhline(0.0, color="#111111", lw=2)
    axes[1].set_ylabel("δ / σ", fontsize=16, fontweight="bold")
    axes[1].set_title("Нормированная поправка: чем дальше от 0, тем сильнее метод исправил поток", fontsize=20, fontweight="bold")
    axes[1].set_xticks(indices)
    axes[1].set_xticklabels(ids, rotation=35, fontsize=14, fontweight="bold")
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    handles = [
        mpatches.Patch(color="#dc2626", label="Внесенная грубая ошибка"),
        mpatches.Patch(color="#2563eb", label="Обычный поток"),
    ]
    axes[0].legend(handles=handles, fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=PLOT_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_relative_error(
    net: Mapping[str, object],
    x_meas: np.ndarray,
    x_rec: np.ndarray,
    x_sicc: np.ndarray | None,
    filename: Path,
) -> None:
    ids = stream_ids(net)
    x_true = np.asarray(net["x_true"], dtype=float)
    denominator = np.maximum(np.abs(x_true), ZERO_DIVISION_EPS)
    rel_meas = 100.0 * np.abs(x_meas - x_true) / denominator
    rel_nested = 100.0 * np.abs(x_rec - x_true) / denominator
    rel_sicc = 100.0 * np.abs(x_sicc - x_true) / denominator if x_sicc is not None else None
    indices = np.arange(len(ids))
    width = 0.24

    fig, ax = plt.subplots(figsize=(22, 9))
    ax.bar(indices - width, rel_meas, width, label="Измерения", color="#dc2626", alpha=0.85)
    ax.bar(indices, rel_nested, width, label="Тихонов", color="#2563eb", alpha=0.9)
    if rel_sicc is not None:
        ax.bar(indices + width, rel_sicc, width, label="SICC", color="#9333ea", alpha=0.85)
    ax.set_xticks(indices)
    ax.set_xticklabels(ids, rotation=35, fontsize=14, fontweight="bold")
    ax.set_ylabel("Относительная ошибка, %", fontsize=17, fontweight="bold")
    ax.set_title("Относительная ошибка по потокам", fontsize=22, fontweight="bold")
    ax.legend(fontsize=15)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(filename, dpi=PLOT_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_analytics_png(
    net: Mapping[str, object],
    x_meas: np.ndarray,
    nested: RegularizedBalanceResult,
    x_sicc: np.ndarray | None,
    gross_errors: Mapping[str, float],
    filename: Path,
) -> None:
    ids = stream_ids(net)
    streams = list(net["streams"])  # type: ignore[arg-type]
    x_true = np.asarray(net["x_true"], dtype=float)
    std_devs = np.asarray(net["std_devs"], dtype=float)
    incidence = np.asarray(net["A_internal"], dtype=float)
    rows: list[list[str]] = []

    for index, stream_id in enumerate(ids):
        measured_rel = 100.0 * abs(x_meas[index] - x_true[index]) / max(abs(x_true[index]), ZERO_DIVISION_EPS)
        nested_rel = 100.0 * abs(nested.x[index] - x_true[index]) / max(abs(x_true[index]), ZERO_DIVISION_EPS)
        sicc_rel = 100.0 * abs(x_sicc[index] - x_true[index]) / max(abs(x_true[index]), ZERO_DIVISION_EPS) if x_sicc is not None else np.nan
        gross_error = gross_errors.get(stream_id, 0.0)
        row = [
            stream_id,
            wrap_table_text(streams[index]["name"], ANALYTICS_NAME_WRAP_WIDTH),
            f"{gross_error:+.4f}",
            f"{x_true[index]:.3f}",
            f"{std_devs[index]:.4f}",
            f"{x_meas[index]:.3f}",
            f"{measured_rel:.2f}%",
            f"{nested.x[index]:.3f}",
            f"{nested_rel:.2f}%",
        ]
        if x_sicc is not None:
            row.append(f"{x_sicc[index]:.3f}")
            row.append(f"{sicc_rel:.2f}%")
        row.extend(
            [
            f"{nested.x[index] - x_meas[index]:+.3f}",
            f"{normalized_delta(nested.x, x_meas, std_devs)[index]:+.2f}",
            ]
        )
        rows.append(row)

    columns = ["ID", "Название", "Внес. грубая ошибка, т/ч", "Истина", "σ", "Изм.", "Ошибка изм., %", "Тихонов", "Ошибка Тих., %"]
    if x_sicc is not None:
        columns.extend(["SICC", "Ошибка SICC, %"])
    columns.extend(["δ", "δ/σ"])

    fig, ax = plt.subplots(figsize=(34, 13.5))
    ax.axis("off")
    ax.set_title("Аналитика по потокам: внесенные ошибки, Тихонов и SICC", fontsize=24, fontweight="bold", pad=12)
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        bbox=ANALYTICS_TABLE_BBOX,
        colWidths=ANALYTICS_COLUMN_WIDTHS_WITH_SICC if x_sicc is not None else None,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1.0, 2.15)
    for column in range(len(columns)):
        table[0, column].set_facecolor("#1f2937")
        table[0, column].set_text_props(color="white", fontweight="bold")
    for row_index in range(len(rows)):
        bg = "#f8fafc" if row_index % 2 == 0 else "white"
        for column in range(len(columns)):
            table[row_index + 1, column].set_facecolor(bg)
        if abs(float(rows[row_index][2])) > ZERO_DIVISION_EPS:
            table[row_index + 1, 2].set_facecolor("#dc2626")
            table[row_index + 1, 2].set_text_props(color="white", fontweight="bold")
        delta_sigma_index = len(columns) - 1
        if abs(float(rows[row_index][delta_sigma_index])) >= IDENTIFICATION_Z_THRESHOLD:
            table[row_index + 1, delta_sigma_index].set_facecolor("#fecaca")
            table[row_index + 1, delta_sigma_index].set_text_props(color="#991b1b", fontweight="bold")

    nested_balance_abs = float(np.max(np.abs(incidence.dot(nested.x))))
    sicc_balance_abs = float(np.max(np.abs(incidence.dot(x_sicc)))) if x_sicc is not None else np.nan
    footer = (
        f"λ={nested.lambda_reg:.4e}; остаток баланса Тихонов={nested_balance_abs:.4f} т/ч; "
        f"остаток баланса SICC={sicc_balance_abs:.4f} т/ч. "
        "Меньше остаток - лучше сходится материальный баланс."
    )
    explanation = (
        "Сноска:\n"
        "Эта таблица - один тестовый запуск; красных потоков с внесенной ошибкой здесь ровно 5: P1, P6, Q1, Q3, R1.\n"
        "В Monte Carlo красный бокс означает целый сценарий с ошибкой, поэтому там красных боксов больше.\n"
        "Внес. грубая ошибка - специально добавленная ошибка, т/ч; красная ячейка = ошибка была внесена.\n"
        "Если 0.0000, то грубую ошибку в этот поток не добавляли.\n"
        "Числа заданы через погрешность датчика: например P1 = 8σ = 8 * 0.0550 = 0.4400 т/ч.\n"
        "Истина - правильное значение; σ - погрешность датчика; Изм. - показание датчика; Ошибка изм. - ошибка измерения.\n"
        "Тихонов - результат регуляризации; Ошибка Тих. - ошибка Тихонова; "
        "SICC (Serial Identification with Collective Compensation) - другой метод сведения; Ошибка SICC - ошибка SICC.\n"
        "δ - поправка Тихонова; δ/σ - во сколько раз поправка больше погрешности датчика."
    )
    fig.text(
        0.08,
        0.085,
        explanation,
        ha="left",
        va="bottom",
        fontsize=12.5,
        color="#111827",
        bbox=dict(facecolor="#f8fafc", edgecolor="#94a3b8", boxstyle="round,pad=0.6"),
    )
    fig.text(0.5, 0.025, footer, ha="center", fontsize=13, color="#334155")
    plt.tight_layout()
    plt.savefig(filename, dpi=PLOT_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_stream_names_png(
    net: Mapping[str, object],
    gross_errors: Mapping[str, float],
    filename: Path,
) -> None:
    rows = [
        [
            str(stream["id"]),
            wrap_table_text(stream["name"], STREAM_NAMES_WRAP_WIDTH),
            "да" if str(stream["id"]) in gross_errors else "нет",
        ]
        for stream in list(net["streams"])  # type: ignore[arg-type]
    ]
    fig, ax = plt.subplots(figsize=(15, 11))
    ax.axis("off")
    ax.set_title("Названия потоков на схеме", fontsize=24, fontweight="bold", pad=12)
    table = ax.table(
        cellText=rows,
        colLabels=["ID", "Название потока", "Внесена грубая ошибка"],
        cellLoc="center",
        bbox=[0.02, 0.04, 0.96, 0.88],
        colWidths=STREAM_NAMES_COLUMN_WIDTHS,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(15)
    table.scale(1.0, 2.1)
    for column in range(3):
        table[0, column].set_facecolor("#1f2937")
        table[0, column].set_text_props(color="white", fontweight="bold")
    for row_index, row in enumerate(rows, start=1):
        table[row_index, 0].set_text_props(fontweight="bold")
        table[row_index, 1].set_text_props(ha="left")
        if row[2] == "да":
            table[row_index, 2].set_facecolor("#dc2626")
            table[row_index, 2].set_text_props(color="white", fontweight="bold")
        elif row_index % 2 == 1:
            for column in range(3):
                table[row_index, column].set_facecolor("#f8fafc")
    plt.savefig(filename, dpi=PLOT_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_single_run_csv(
    net: Mapping[str, object],
    run: PlantRunResult,
    gross_errors: Mapping[str, float],
    filename: Path,
) -> None:
    x_true = np.asarray(net["x_true"], dtype=float)
    std_devs = np.asarray(net["std_devs"], dtype=float)
    with filename.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["id", "true", "sigma", "measured", "lambda_result", "delta", "delta_sigma", "relative_error_pct", "gross_error"])
        for index, stream_id in enumerate(stream_ids(net)):
            relative_error = 100.0 * abs(run.nested.x[index] - x_true[index]) / max(abs(x_true[index]), ZERO_DIVISION_EPS)
            writer.writerow(
                [
                    stream_id,
                    f"{x_true[index]:.8f}",
                    f"{std_devs[index]:.8f}",
                    f"{run.x_meas[index]:.8f}",
                    f"{run.nested.x[index]:.8f}",
                    f"{run.nested.x[index] - run.x_meas[index]:.8f}",
                    f"{normalized_delta(run.nested.x, run.x_meas, std_devs)[index]:.8f}",
                    f"{relative_error:.8f}",
                    f"{gross_errors.get(stream_id, 0.0):.8f}",
                ]
            )


def build_monte_carlo_scenarios(net: Mapping[str, object]) -> list[tuple[str, dict[str, float]]]:
    std_devs = np.asarray(net["std_devs"], dtype=float)
    indices = stream_indices(net)
    target_streams = ["P1", "Q1", "R1", "P6", "Q3"]
    sigma_levels = [2.0, 5.0, 10.0]
    scenarios: list[tuple[str, dict[str, float]]] = [("no_gross_error", {})]

    for stream_id in target_streams:
        for level in sigma_levels:
            scenarios.append((f"{stream_id}_{level:.0f}sigma", {stream_id: level * std_devs[indices[stream_id]]}))

    scenarios.append(("combined_current_tph", dict(CURRENT_GROSS_ERRORS)))
    return scenarios


def run_monte_carlo(
    net: Mapping[str, object],
    n_trials: int = DEFAULT_MONTE_CARLO_TRIALS,
    scenarios: Sequence[tuple[str, Mapping[str, float]]] | None = None,
    output_dir: Path = RESULTS_DIR,
    pilot_points: int = MONTE_CARLO_PILOT_POINTS,
) -> list[MonteCarloRow]:
    scenario_list = list(scenarios) if scenarios is not None else build_monte_carlo_scenarios(net)
    x_true = np.asarray(net["x_true"], dtype=float)
    std_devs = np.asarray(net["std_devs"], dtype=float)
    rows: list[MonteCarloRow] = []
    base_trials_per_scenario = n_trials // len(scenario_list)
    extra_trials = n_trials % len(scenario_list)

    for scenario_index, (scenario_name, gross_errors) in enumerate(scenario_list):
        scenario_trials = base_trials_per_scenario + (1 if scenario_index < extra_trials else 0)
        for trial in range(scenario_trials):
            seed = scenario_index * 100_000 + trial
            x_meas = generate_measurements(net, seed=seed, gross_errors=gross_errors)
            lambda_result = solve_nested_for_measurements(net, x_meas, pilot_points=pilot_points)
            nested_x = lambda_result.best_result.x
            sicc = solve_sicc(net, x_meas)
            sicc_x = sicc.x
            nested_detected = {stream_ids(net)[index] for index, score in enumerate(np.abs(normalized_delta(nested_x, x_meas, std_devs))) if score >= IDENTIFICATION_Z_THRESHOLD}
            sicc_detected = set(sicc.bias_map.keys())

            rows.append(
                MonteCarloRow(
                    scenario=scenario_name,
                    trial=seed,
                    lambda_reg=lambda_result.best_lambda,
                    nested_rmse=root_mean_square_error(nested_x, x_true),
                    nested_relative_mae=relative_mae(nested_x, x_true),
                    measurement_rmse=root_mean_square_error(x_meas, x_true),
                    measurement_relative_mae=relative_mae(x_meas, x_true),
                    sicc_rmse=root_mean_square_error(sicc_x, x_true) if sicc_x is not None else np.nan,
                    sicc_relative_mae=relative_mae(sicc_x, x_true) if sicc_x is not None else np.nan,
                    nested_detection_rate=true_positive_rate(nested_detected, gross_errors),
                    sicc_detection_rate=true_positive_rate(sicc_detected, gross_errors),
                )
            )

    write_monte_carlo_csv(rows, output_dir / "plant_monte_carlo.csv")
    write_monte_carlo_summary_csv(rows, output_dir / "plant_scenario_summary.csv")
    plot_error_histograms(rows, output_dir / "plant_error_histograms.png")
    plot_monte_carlo_summary(rows, output_dir / "plant_monte_carlo_summary.png")
    return rows


def write_monte_carlo_csv(rows: Sequence[MonteCarloRow], filename: Path) -> None:
    fields = [
        ("scenario", "scenario"),
        ("trial", "trial"),
        ("lambda_reg", "lambda"),
        ("nested_rmse", "lambda_rmse"),
        ("nested_relative_mae", "lambda_relative_mae"),
        ("measurement_rmse", "measurement_rmse"),
        ("measurement_relative_mae", "measurement_relative_mae"),
        ("sicc_rmse", "sicc_rmse"),
        ("sicc_relative_mae", "sicc_relative_mae"),
        ("nested_detection_rate", "lambda_detection_rate"),
        ("sicc_detection_rate", "sicc_detection_rate"),
    ]
    with filename.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([header for _, header in fields])
        for row in rows:
            writer.writerow([getattr(row, field) for field, _ in fields])


def scenario_groups(rows: Sequence[MonteCarloRow]) -> dict[str, list[MonteCarloRow]]:
    grouped: dict[str, list[MonteCarloRow]] = {}
    for row in rows:
        grouped.setdefault(row.scenario, []).append(row)
    return grouped


def write_monte_carlo_summary_csv(rows: Sequence[MonteCarloRow], filename: Path) -> None:
    grouped = scenario_groups(rows)
    with filename.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["scenario", "n", "lambda_mean", "lambda_rmse_mean", "meas_rmse_mean", "sicc_rmse_mean", "lambda_detection_mean", "sicc_detection_mean"])
        for scenario, scenario_rows in grouped.items():
            writer.writerow(
                [
                    scenario,
                    len(scenario_rows),
                    f"{np.mean([row.lambda_reg for row in scenario_rows]):.8f}",
                    f"{np.mean([row.nested_rmse for row in scenario_rows]):.8f}",
                    f"{np.mean([row.measurement_rmse for row in scenario_rows]):.8f}",
                    f"{np.nanmean([row.sicc_rmse for row in scenario_rows]):.8f}",
                    f"{np.mean([row.nested_detection_rate for row in scenario_rows]):.8f}",
                    f"{np.mean([row.sicc_detection_rate for row in scenario_rows]):.8f}",
                ]
            )


def is_gross_error_scenario(scenario: str) -> bool:
    return scenario != "no_gross_error"


def format_scenario_label(scenario: str) -> str:
    if scenario == "no_gross_error":
        return "без\nгрубой"
    if scenario == "combined_current_tph":
        return "5 ошибок\nт/ч"
    if scenario.endswith("sigma"):
        stream_id, sigma_value = scenario.removesuffix("sigma").split("_", maxsplit=1)
        sigma_int = int(float(sigma_value))
        sigma_word = "сигмы" if sigma_int == 2 else "сигм"
        return f"{stream_id}\n{sigma_int} {sigma_word}"
    return scenario.replace("_", "\n")


def plot_error_histograms(rows: Sequence[MonteCarloRow], filename: Path) -> None:
    nested_rmse = np.array([row.nested_rmse for row in rows], dtype=float)
    measured_rmse = np.array([row.measurement_rmse for row in rows], dtype=float)
    sicc_rmse = np.array([row.sicc_rmse for row in rows], dtype=float)
    nested_rel = 100.0 * np.array([row.nested_relative_mae for row in rows], dtype=float)
    measured_rel = 100.0 * np.array([row.measurement_relative_mae for row in rows], dtype=float)
    sicc_rel = 100.0 * np.array([row.sicc_relative_mae for row in rows], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    axes[0].hist(measured_rmse, bins=40, alpha=0.55, color="#dc2626", label="Измерения")
    axes[0].hist(nested_rmse, bins=40, alpha=0.65, color="#2563eb", label="Тихонов")
    axes[0].hist(sicc_rmse[np.isfinite(sicc_rmse)], bins=40, alpha=0.50, color="#9333ea", label="SICC")
    axes[0].set_title("RMSE для разных ошибок", fontsize=20, fontweight="bold")
    axes[0].set_xlabel("RMSE, т/ч", fontsize=16)
    axes[0].set_ylabel("Количество прогонов", fontsize=16)
    axes[0].legend(fontsize=14)
    axes[0].grid(axis="y", alpha=0.35)

    axes[1].hist(measured_rel, bins=40, alpha=0.55, color="#dc2626", label="Измерения")
    axes[1].hist(nested_rel, bins=40, alpha=0.65, color="#2563eb", label="Тихонов")
    axes[1].hist(sicc_rel[np.isfinite(sicc_rel)], bins=40, alpha=0.50, color="#9333ea", label="SICC")
    axes[1].set_title("Относительная ошибка для разных ошибок", fontsize=20, fontweight="bold")
    axes[1].set_xlabel("Средняя относительная ошибка, %", fontsize=16)
    axes[1].legend(fontsize=14)
    axes[1].grid(axis="y", alpha=0.35)

    plt.tight_layout()
    plt.savefig(filename, dpi=PLOT_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_monte_carlo_summary(rows: Sequence[MonteCarloRow], filename: Path) -> None:
    grouped = scenario_groups(rows)
    labels = list(grouped.keys())
    display_labels = [format_scenario_label(label) for label in labels]
    nested_rmse = [[row.nested_rmse for row in grouped[label]] for label in labels]
    lambda_values = [[row.lambda_reg for row in grouped[label]] for label in labels]
    nested_detection = [np.mean([row.nested_detection_rate for row in grouped[label]]) for label in labels]
    sicc_detection = [np.mean([row.sicc_detection_rate for row in grouped[label]]) for label in labels]

    fig, axes = plt.subplots(2, 1, figsize=(24, 16))
    boxplot = axes[0].boxplot(tick_labels=display_labels, x=nested_rmse, patch_artist=True, medianprops=dict(color="#111827", lw=2.8))
    for index, patch in enumerate(boxplot["boxes"]):
        has_gross_error = is_gross_error_scenario(labels[index])
        patch.set_facecolor("#fecaca" if has_gross_error else "#bfdbfe")
        patch.set_edgecolor("#dc2626" if has_gross_error else "#1d4ed8")
        patch.set_linewidth(2.2)
    for index, label in enumerate(labels, start=1):
        if is_gross_error_scenario(label):
            axes[0].axvspan(index - 0.45, index + 0.45, color="#fee2e2", alpha=0.22, zorder=0)
    axes[0].set_title("Monte Carlo: RMSE λ по сценариям", fontsize=22, fontweight="bold")
    axes[0].set_ylabel("RMSE, т/ч", fontsize=17)
    axes[0].tick_params(axis="x", rotation=45, labelsize=12)
    axes[0].grid(axis="y", alpha=0.35)
    axes[0].text(
        0.985,
        0.95,
        "Красный бокс = весь сценарий с внесенной ошибкой,\nа не отдельный поток.",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=14,
        color="#7f1d1d",
        bbox=dict(facecolor="white", edgecolor="#dc2626", boxstyle="round,pad=0.45", alpha=0.94),
    )
    axes[0].legend(
        handles=[
            mpatches.Patch(facecolor="#bfdbfe", edgecolor="#1d4ed8", label="Сценарий без ошибки"),
            mpatches.Patch(facecolor="#fecaca", edgecolor="#dc2626", label="Сценарий с внесенной ошибкой"),
        ],
        fontsize=14,
        loc="upper left",
    )

    x_pos = np.arange(len(labels))
    width = 0.38
    axes[1].bar(x_pos - width / 2.0, nested_detection, width, label="λ", color="#2563eb")
    axes[1].bar(x_pos + width / 2.0, sicc_detection, width, label="SICC", color="#9333ea")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(display_labels, rotation=45, ha="right", fontsize=12)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_ylabel("Доля верной идентификации", fontsize=17)
    axes[1].set_title(f"Как часто метод нашел внесенную ошибку: медиана λ={np.median([value for group in lambda_values for value in group]):.2e}", fontsize=22, fontweight="bold")
    axes[1].legend(fontsize=15)
    axes[1].grid(axis="y", alpha=0.35)

    plt.tight_layout()
    plt.savefig(filename, dpi=PLOT_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def create_all_single_case_outputs(
    net: Mapping[str, object],
    run: PlantRunResult,
    gross_errors: Mapping[str, float],
    output_dir: Path,
) -> None:
    x_sicc = run.sicc.x
    plot_lambda_optimization(run.lambda_result, output_dir / "plant_lambda_optimization.png")
    plot_plant_graph_readable(net, run.x_meas, run.nested.x, gross_errors, output_dir / "plant_graph_readable.png")
    plot_plant_graph_readable(net, run.x_meas, run.nested.x, gross_errors, Path("plant_graph.png"))
    plot_corridor_readable(net, run.x_meas, run.nested.x, x_sicc, output_dir / "plant_corridor_readable.png")
    plot_delta(net, run.x_meas, run.nested.x, gross_errors, output_dir / "plant_delta.png")
    plot_relative_error(net, run.x_meas, run.nested.x, x_sicc, output_dir / "plant_relative_error.png")
    save_analytics_png(net, run.x_meas, run.nested, x_sicc, gross_errors, output_dir / "plant_analytics_readable.png")
    save_stream_names_png(net, gross_errors, output_dir / "plant_stream_names.png")
    write_single_run_csv(net, run, gross_errors, output_dir / "plant_single_run.csv")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

    output_dir = ensure_results_dir()
    net = define_plant_network()
    balance_check = np.asarray(net["A_internal"], dtype=float).dot(np.asarray(net["x_true"], dtype=float))
    print("=" * 80)
    print("СХЕМА ЦЕХА: регуляризация Тихонова с подбором λ")
    print(f"Баланс true по внутренним узлам: max|Ax|={np.max(np.abs(balance_check)):.8f}")
    print("=" * 80)

    run = run_single_case(net, CURRENT_GROSS_ERRORS, seed=42, pilot_points=MAIN_PILOT_POINTS)
    print(f"λ* = {run.lambda_result.best_lambda:.8e}")
    print(f"Inner success: {run.nested.success}; {run.nested.message}")
    print(f"Balance term = {run.nested.balance_term:.6f}")
    print(f"Delta term = {run.nested.delta_term:.6f}")
    print(f"SICC: {run.sicc.status}; biases={run.sicc.bias_map}")
    create_all_single_case_outputs(net, run, CURRENT_GROSS_ERRORS, output_dir)

    mc_trials = int(os.environ.get("PLANT_MC_TRIALS", str(DEFAULT_MONTE_CARLO_TRIALS)))
    print(f"Monte Carlo: {mc_trials} прогонов всего")
    run_monte_carlo(net, n_trials=mc_trials, output_dir=output_dir)

    print("\nСохранены результаты:")
    for path in sorted(output_dir.iterdir()):
        print(f"  -> {path}")


if __name__ == "__main__":
    main()
