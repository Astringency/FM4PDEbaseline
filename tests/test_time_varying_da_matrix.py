from __future__ import annotations

from baselines.capabilities import resolve_capability


def test_4dvar_and_vivid_only_main_with_time_varying_full_trajectory():
    var4d = resolve_capability(
        "var4d",
        "reaction_diffusion",
        "sparse_solution",
        "time_varying",
        "time_varying",
        load_full_trajectory=True,
    )
    assert var4d.support_status == "native"
    assert var4d.paper_table_eligible is True

    endpoint = resolve_capability(
        "var4d",
        "reaction_diffusion",
        "sparse_solution",
        "time_varying",
        "time_varying",
        load_full_trajectory=False,
    )
    assert endpoint.support_status == "adapted"
    assert endpoint.paper_table_eligible is False

    vivid = resolve_capability(
        "vivid",
        "shallow_water",
        "sparse_solution",
        "time_varying",
        "time_varying",
        load_full_trajectory=True,
        train_inverse_operator=True,
    )
    assert vivid.support_status == "official_adapter"
    assert vivid.paper_table_eligible is True
