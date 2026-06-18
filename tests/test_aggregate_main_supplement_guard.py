from __future__ import annotations

from baselines.aggregate_results import partition_rows_for_tables


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


def test_vivid_style_true_flag_is_downgraded_to_supplement():
    row = _row(
        baseline="vivid",
        capability_status="official_adapter",
        implementation_required="official",
        implementation_mode_effective="official_architecture",
        eligible_implementation_modes='["official"]',
        adapter_status="vivid_style_trained_inverse_operator",
    )
    main, supplement = partition_rows_for_tables([row])
    assert main == []
    assert len(supplement) == 1
    assert supplement[0]["paper_table_eligible"] is False
    assert "downgraded_to_supplement" in supplement[0]["aggregation_warning"]


def test_fallback_true_flag_is_downgraded_to_supplement():
    row = _row(fallback_used=True, adapter_status="official_code")
    main, supplement = partition_rows_for_tables([row])
    assert main == []
    assert len(supplement) == 1
    assert "fallback_used=true" in supplement[0]["aggregation_warning"]


def test_canonical_math_pde_opt_stays_main():
    row = _row(baseline="pde_opt", capability_status="native", adapter_status="canonical_math")
    main, supplement = partition_rows_for_tables([row])
    assert main == [row]
    assert supplement == []
