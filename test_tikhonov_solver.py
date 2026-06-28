from __future__ import annotations

import numpy as np

from main_plant import (
    CURRENT_GROSS_ERRORS,
    create_all_single_case_outputs,
    define_plant_network,
    generate_measurements,
    run_monte_carlo,
    run_single_case,
)
from tikhonov_solver import optimize_lambda_nested, solve_regularized_balance


def test_define_plant_network_true_balance() -> None:
    net = define_plant_network()

    balance = np.asarray(net["A_internal"], dtype=float).dot(np.asarray(net["x_true"], dtype=float))

    assert np.max(np.abs(balance)) < 1e-9
    assert np.asarray(net["A_internal"]).shape == (3, 14)
    assert len(net["streams"]) == 14
    assert len(net["std_devs"]) == 14


def test_inner_solver_returns_finite_metrics() -> None:
    incidence = np.array([[1.0, -1.0]])
    measurements = np.array([10.0, 8.0])
    std_devs = np.array([1.0, 1.0])

    result = solve_regularized_balance(
        incidence,
        measurements,
        std_devs,
        lambda_reg=1.0,
        bounds=[(0.0, None), (0.0, None)],
    )

    assert result.success
    assert np.all(np.isfinite(result.x))
    assert np.isfinite(result.balance_term)
    assert np.isfinite(result.delta_term)
    assert np.isfinite(result.objective)


def test_objective_penalizes_delta_not_flow_size() -> None:
    incidence = np.array([[1.0, -1.0]])
    measurements = np.array([10.0, 10.0])
    std_devs = np.array([1.0, 1.0])

    result = solve_regularized_balance(
        incidence,
        measurements,
        std_devs,
        lambda_reg=100.0,
        bounds=[(0.0, None), (0.0, None)],
    )

    assert result.success
    assert np.allclose(result.x, measurements, atol=1e-5)


def test_lambda_optimizer_selects_lcurve_corner() -> None:
    net = define_plant_network()
    x_meas = generate_measurements(net, seed=7, gross_errors={"P1": 1.0})

    result = optimize_lambda_nested(
        np.asarray(net["A_internal"], dtype=float),
        x_meas,
        np.asarray(net["std_devs"], dtype=float),
        lambda_bounds=(1e-6, 1e4),
        bounds=[(0.0, None) for _ in range(len(x_meas))],
        pilot_points=8,
    )

    assert 1e-6 <= result.best_lambda <= 1e4
    assert np.isclose(result.best_lambda, result.lcurve_best_lambda)
    assert result.best_result.success
    assert result.pilot["lambdas"].size == 8
    assert result.pilot["lcurve_curvatures"].size == 8
    assert np.isfinite(result.pilot["lcurve_curvatures"]).all()


def test_monte_carlo_uses_optimized_lambda(tmp_path) -> None:
    net = define_plant_network()
    scenarios = [("quick", {"P1": 1.0})]

    rows = run_monte_carlo(
        net,
        n_trials=2,
        scenarios=scenarios,
        output_dir=tmp_path,
        pilot_points=6,
    )

    lambdas = np.array([row.lambda_reg for row in rows])
    assert len(rows) == 2
    assert np.all(np.isfinite(lambdas))
    assert not np.allclose(lambdas, 0.01)
    assert (tmp_path / "plant_monte_carlo.csv").exists()
    assert (tmp_path / "plant_error_histograms.png").exists()
    assert (tmp_path / "plant_monte_carlo_summary.png").exists()


def test_single_case_plots_created(tmp_path) -> None:
    net = define_plant_network()
    run = run_single_case(net, CURRENT_GROSS_ERRORS, seed=42, pilot_points=8)

    create_all_single_case_outputs(net, run, CURRENT_GROSS_ERRORS, tmp_path)

    expected_files = {
        "plant_lambda_optimization.png",
        "plant_graph_readable.png",
        "plant_corridor_readable.png",
        "plant_delta.png",
        "plant_relative_error.png",
        "plant_analytics_readable.png",
        "plant_single_run.csv",
    }
    existing_files = {path.name for path in tmp_path.iterdir()}
    assert expected_files <= existing_files
