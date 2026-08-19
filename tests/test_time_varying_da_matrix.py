from __future__ import annotations

from baselines.capabilities import resolve_capability


def test_burger_var4d_and_vivid_are_explicitly_adapted_not_official():
    var4d = resolve_capability(
        "var4d",
        "burger",
        "sparse_solution",
        "random_per_sample",
        "time_varying_da_main",
        load_full_trajectory=True,
    )
    vivid = resolve_capability(
        "vivid",
        "burger",
        "sparse_solution",
        "random_per_sample",
        "time_varying_da_main",
        load_full_trajectory=True,
        train_inverse_operator=True,
        uses_official_inverse_observation_operator=False,
    )

    for capability in (var4d, vivid):
        assert capability.support_status == "adapted"
        assert capability.implementation_required == "adapted_allowed"
        assert capability.paper_table_eligible is False
        assert capability.official_native_eligible is False


def test_burger_variational_adapters_require_full_trajectory():
    for baseline in ("var4d", "vivid"):
        capability = resolve_capability(
            baseline,
            "burger",
            "sparse_solution",
            "random_per_sample",
            "time_varying_da_main",
            load_full_trajectory=False,
        )
        assert capability.support_status == "unsupported"
        assert capability.paper_table_eligible is False


def test_removed_non_burgers_variational_da_entries_are_unsupported():
    for baseline in ("var4d", "vivid"):
        for pde in ("nsnonbounded", "reaction_diffusion", "shallow_water"):
            capability = resolve_capability(
                baseline,
                pde,
                "sparse_solution",
                "time_varying",
                "time_varying",
                load_full_trajectory=True,
            )
            assert capability.support_status == "unsupported"
            assert "only to Burgers" in capability.reason
