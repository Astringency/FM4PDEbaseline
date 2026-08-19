import numpy as np
import pytest

torch = pytest.importorskip("torch")
scipy_interpolate = pytest.importorskip("scipy.interpolate")

from baselines.common.physics import (
    burgers_residual,
    darcy_residual,
    helmholtz_inverse_residual,
    physics_losses,
)


def _cubic_matrix(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    return scipy_interpolate.CubicSpline(source, np.eye(source.size), axis=0)(target)


def test_darcy_residual_exactly_reconstructs_matlab_generator_grid():
    n = 12
    cell = (np.arange(n) + 0.5) / n
    nodal = np.linspace(0.0, 1.0, n)
    c2n = _cubic_matrix(cell, nodal)
    n2c = _cubic_matrix(nodal, cell)
    rng = np.random.default_rng(17)
    coeff_cell = np.where(rng.standard_normal((n, n)) > 0.0, 12.0, 4.0)
    coeff_nodal = c2n @ coeff_cell @ c2n.T
    h = 1.0 / (n - 1)
    size = (n - 2) ** 2
    matrix = np.zeros((size, size))

    def idx(i: int, j: int) -> int:
        return (i - 1) * (n - 2) + j - 1

    for i in range(1, n - 1):
        for j in range(1, n - 1):
            row = idx(i, j)
            for ni, nj in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                face = 0.5 * (coeff_nodal[i, j] + coeff_nodal[ni, nj]) / h**2
                matrix[row, row] += face
                if 0 < ni < n - 1 and 0 < nj < n - 1:
                    matrix[row, idx(ni, nj)] -= face
    solution_nodal = np.zeros((n, n))
    solution_nodal[1:-1, 1:-1] = np.linalg.solve(matrix, np.ones(size)).reshape(n - 2, n - 2)
    solution_cell = n2c @ solution_nodal @ n2c.T
    residual = darcy_residual(
        torch.tensor(coeff_cell, dtype=torch.float64).reshape(1, 1, n, n),
        torch.tensor(solution_cell, dtype=torch.float64).reshape(1, 1, n, n),
    )
    assert residual.square().mean().sqrt().item() < 1e-8


def test_helmholtz_residual_uses_actual_generator_kronecker_boundary_rows():
    n = 9
    k = 2.0
    h = 1.0 / (n - 1)
    one_d = np.diag(np.full(n, -2.0)) + np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)
    one_d /= h**2
    one_d[0] = 0.0
    one_d[0, 0] = 1.0
    one_d[-1] = 0.0
    one_d[-1, -1] = 1.0
    source = np.random.default_rng(4).standard_normal((n, n))
    rhs = source.copy()
    rhs[[0, -1], :] = 0.0
    rhs[:, [0, -1]] = 0.0
    system = np.kron(np.eye(n), one_d) + np.kron(one_d, np.eye(n)) + k**2 * np.eye(n * n)
    solution = np.linalg.solve(system, rhs.reshape(-1, order="F")).reshape(n, n, order="F")
    residual = helmholtz_inverse_residual(
        torch.tensor(source, dtype=torch.float64).reshape(1, 1, n, n),
        torch.tensor(solution, dtype=torch.float64).reshape(1, 1, n, n),
        k=k,
    )
    assert residual.shape == (1, 1, n, n)
    assert residual.square().mean().sqrt().item() < 1e-8


def test_burgers_full_trajectory_has_128_frames_and_dt_one_over_127():
    n = 128
    times = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
    trajectory = times.reshape(1, 1, n, 1).expand(1, 1, n, n).clone()
    residual = burgers_residual(trajectory, nu=0.0, metadata={"final_time": 1.0})
    assert residual.shape == (1, 1, 126, 128)
    assert torch.allclose(residual, torch.ones_like(residual), atol=1e-12, rtol=1e-12)


def test_steady_heat_boundary_loss_matches_generator_rows_and_corner_precedence():
    n = 6
    coordinate = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
    solution = coordinate.reshape(1, 1, n, 1).expand(1, 1, n, n).clone()
    losses = physics_losses(
        solution,
        "steady_heat_conduction",
        {"source_fields": torch.zeros_like(solution), "u_D": 0.0},
    )
    # 6 bottom + 5 left + 5 right + 4 top-interior rows.  Only the four
    # top-interior raw differences are nonzero; top corners belong to sides.
    assert losses["bc"].item() == pytest.approx(4 * (1.0 / 5.0) ** 2 / 20.0)


def test_endpoint_secant_mode_uses_only_first_and_last_frames():
    trajectory = torch.zeros(1, 1, 5, 8, 8)
    trajectory[:, :, 1:4] = 1000.0
    metadata = {"alpha": 1e-3, "final_time": 1.0, "bc": "periodic", "residual_mode": "endpoint_secant"}
    losses = physics_losses(trajectory, "heat", metadata)
    assert losses["mode"] == "endpoint_secant"
    assert losses["interior"].item() == pytest.approx(0.0)


def test_full_trajectory_mode_rejects_endpoint_only_prediction():
    pred = torch.zeros(1, 1, 8, 8)
    metadata = {
        "input_fields": torch.zeros_like(pred),
        "alpha": 1e-3,
        "final_time": 1.0,
        "bc": "periodic",
        "residual_mode": "full_trajectory_fd",
    }
    with pytest.raises(ValueError, match="full_trajectory_fd"):
        physics_losses(pred, "heat", metadata)
