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
    assert paper_table_eligible(recfno, "official") is True

    pde_opt = resolve_capability("pde_opt", "poisson", "sparse_inverse", "random")
    assert paper_table_eligible(pde_opt, "canonical_math") is True
