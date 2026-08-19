from __future__ import annotations

from baselines.aggregate_results import _latex_rows, aggregate_rows


def _row(**overrides):
    base = {
        "paper_table_eligible": True,
        "fallback_used": False,
        "capability_status": "native",
        "implementation_required": "canonical_math",
        "implementation_mode_effective": "canonical_math",
        "eligible_implementation_modes": '["canonical_math"]',
        "adapter_status": "canonical_math",
        "official_import_success": False,
    }
    base.update(overrides)
    return base


def test_aggregation_keeps_every_completed_result_without_table_tiers():
    rows = [
        _row(baseline="fno", fallback_used=True, relative_l2_solution=1.0),
        _row(baseline="pde_opt", paper_table_eligible=False, relative_l2_solution=2.0),
    ]

    summary = aggregate_rows(rows)

    assert {row["baseline"] for row in summary} == {"fno", "pde_opt"}


def test_latex_rows_report_solution_and_inverse_coefficient_metrics():
    summary = aggregate_rows(
        [
            _row(
                pde="poisson",
                task="inverse",
                baseline="ifno",
                relative_l2_solution=float("nan"),
                relative_l2_input_or_coeff=0.25,
                mse=0.1,
            )
        ]
    )

    latex = _latex_rows(summary)[0]

    assert "relative_l2_solution" in latex
    assert "relative_l2_input_or_coeff" in latex
    assert set(("mae", "obs_mse", "obs_mse_clean", "obs_mse_noisy", "bc_residual", "ic_residual")) <= set(latex)
    assert latex["relative_l2_input_or_coeff_n"] == 1
    assert latex["relative_l2_solution_nan_count"] == 1
