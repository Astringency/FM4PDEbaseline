from __future__ import annotations

from baselines.capabilities import paper_table_eligible, resolve_capability


def test_adapted_and_surrogate_capabilities_are_not_main_eligible():
    fno_inverse = resolve_capability("fno", "poisson", "inverse")
    assert fno_inverse.support_status == "adapted"
    assert paper_table_eligible(fno_inverse, "official") is False

    vivid_style = resolve_capability(
        "vivid",
        "reaction_diffusion",
        "sparse_solution",
        "time_varying",
        "time_varying",
        load_full_trajectory=True,
        train_inverse_operator=False,
    )
    assert vivid_style.support_status == "adapted"
    assert paper_table_eligible(vivid_style, "adapted") is False


def test_native_official_or_canonical_capabilities_are_main_eligible():
    recfno = resolve_capability("recfno", "darcy", "sparse_solution", "random")
    assert paper_table_eligible(
        recfno,
        backend_info={
            "implementation_mode_effective": "official",
            "official_import_success": True,
            "fallback_used": False,
            "adapter_status": "official_code_adapter",
        },
    ) is True

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
            "fallback_used": False,
            "adapter_status": "official_architecture_reimplementation",
        },
    ) is True


def test_vivid_style_without_official_import_is_supplement_only():
    vivid_style = resolve_capability(
        "vivid",
        "reaction_diffusion",
        "sparse_solution",
        "time_varying",
        "time_varying",
        load_full_trajectory=True,
        train_inverse_operator=True,
        uses_official_inverse_observation_operator=False,
    )
    assert vivid_style.support_status == "adapted"
    assert paper_table_eligible(
        vivid_style,
        backend_info={
            "implementation_mode_effective": "adapted",
            "official_import_success": False,
            "fallback_used": False,
            "adapter_status": "vivid_style_trained_inverse_operator",
        },
    ) is False
