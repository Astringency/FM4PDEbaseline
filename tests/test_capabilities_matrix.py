from __future__ import annotations

from baselines.capabilities import resolve_capability


def test_operator_and_sparse_capability_statuses():
    assert resolve_capability("ifno", "darcy", "sparse_inverse", "random").support_status == "unsupported"
    assert resolve_capability("ifno", "darcy", "sparse_solution", "random").support_status == "unsupported"

    fno_sparse = resolve_capability("fno", "poisson", "sparse_solution", "random")
    assert fno_sparse.support_status == "unsupported"
    assert fno_sparse.paper_table_eligible is False

    for baseline in ("recfno", "senseiver", "voronoicnn"):
        cap = resolve_capability(baseline, "darcy", "sparse_solution", "random")
        assert cap.support_status == "adapted"
        assert cap.paper_table_eligible is False
        assert cap.official_native_eligible is False


def test_static_sparse_inverse_physics_baselines_supported():
    for pde in ("poisson", "darcy", "helmholtz", "steady_heat_conduction"):
        assert resolve_capability("pinn_sparse", pde, "sparse_inverse", "random").implementation_required == "canonical_math"
        assert resolve_capability("pde_opt", pde, "sparse_inverse", "random").support_status == "native"


def test_endpoint_da_is_surrogate_not_main():
    cap = resolve_capability("var4d", "heat", "sparse_solution", "random")
    assert cap.support_status == "adapted"
    assert cap.paper_table_eligible is False

    surrogate = resolve_capability("var4d", "heat", "sparse_solution", "time_varying", "time_varying", load_full_trajectory=False)
    assert surrogate.support_status == "adapted"
    assert surrogate.paper_table_eligible is False


def test_sensor_only_protocol_excludes_hidden_truth_and_burger_sparse_inverse():
    for baseline in ("pinn_sparse", "pde_opt"):
        cap = resolve_capability(baseline, "poisson", "sparse_solution", "random_per_sample")
        assert cap.support_status == "unsupported"
        assert "sensor-only" in cap.reason
    for baseline in ("recfno", "senseiver", "voronoicnn", "pinn_sparse", "pde_opt"):
        cap = resolve_capability(baseline, "burger", "sparse_inverse", "random_per_sample")
        assert cap.support_status == "unsupported"
