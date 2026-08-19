from __future__ import annotations

from baselines.capabilities import paper_table_eligible, resolve_capability


def test_adapted_capabilities_and_vivid_without_a_trained_official_inverse_are_not_main_eligible():
    fno_inverse = resolve_capability("fno", "poisson", "inverse")
    assert fno_inverse.support_status == "adapted"
    assert paper_table_eligible(fno_inverse, "official") is False

    vivid_style = resolve_capability(
        "vivid",
        "burger",
        "sparse_solution",
        "time_varying",
        "time_varying",
        load_full_trajectory=True,
        train_inverse_operator=False,
    )
    assert vivid_style.support_status == "unsupported"
    assert paper_table_eligible(vivid_style, "official_architecture") is False


def test_unified_component_adapter_is_not_misreported_as_official_native():
    recfno = resolve_capability("recfno", "darcy", "sparse_solution", "random")
    assert paper_table_eligible(
        recfno,
        backend_info={
            "implementation_mode_effective": "official",
            "official_import_success": True,
            "fallback_used": False,
            "adapter_status": "official_code_adapter",
        },
    ) is False
    assert recfno.official_native_eligible is False

    pde_opt = resolve_capability("pde_opt", "poisson", "sparse_inverse", "random")
    assert paper_table_eligible(pde_opt, "canonical_math") is True


def test_official_requirement_is_not_satisfied_by_canonical_math_or_fallback():
    fno = resolve_capability("fno", "poisson", "forward")
    assert paper_table_eligible(fno, "canonical_math") is False
    assert paper_table_eligible(
        fno,
        backend_info={
            "implementation_mode_effective": "official",
            "official_import_success": False,
            "fallback_used": False,
            "adapter_status": "official_code",
        },
    ) is False
    assert paper_table_eligible(
        fno,
        backend_info={
            "implementation_mode_effective": "official",
            "official_import_success": True,
            "fallback_used": True,
            "adapter_status": "official_code",
        },
    ) is False


def test_official_architecture_must_be_explicitly_allowed():
    recfno = resolve_capability("recfno", "darcy", "sparse_solution", "random")
    assert paper_table_eligible(
        recfno,
        backend_info={
            "implementation_mode_effective": "official_architecture",
            "official_import_success": False,
            "official_reimplementation_success": False,
            "fallback_used": False,
            "adapter_status": "official_architecture_reimplementation",
        },
    ) is False

    voronoicnn = resolve_capability("voronoicnn", "darcy", "sparse_solution", "random")
    assert paper_table_eligible(
        voronoicnn,
        backend_info={
            "implementation_mode_effective": "adapted",
            "implementation_source": "recfno_unet_as_voronoi_cnn_adaptation",
            "official_import_success": True,
            "fallback_used": True,
            "adapter_status": "fallback_recfno_unet_adaptation",
        },
    ) is False
    assert paper_table_eligible(
        voronoicnn,
        backend_info={
            "implementation_mode_effective": "official_architecture",
            "official_import_success": False,
            "official_reimplementation_success": True,
            "fallback_used": False,
            "adapter_status": "official_architecture_reimplementation",
        },
    ) is False


def test_vivid_official_architecture_adapter_is_table_eligible_but_not_official_native():
    vivid = resolve_capability(
        "vivid",
        "burger",
        "sparse_solution",
        "time_varying",
        "time_varying",
        load_full_trajectory=True,
        train_inverse_operator=True,
        uses_official_inverse_observation_operator=True,
    )
    assert vivid.support_status == "official_adapter"
    assert paper_table_eligible(
        vivid,
        backend_info={
            "implementation_mode_effective": "official_architecture",
            "official_import_success": False,
            "official_reimplementation_success": True,
            "fallback_used": False,
            "adapter_status": "official_architecture_vivid_burgers_task_adapter",
        },
    ) is True
    assert vivid.official_native_eligible is False
