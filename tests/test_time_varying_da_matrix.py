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

    vivid_style = resolve_capability(
        "vivid",
        "shallow_water",
        "sparse_solution",
        "time_varying",
        "time_varying",
        load_full_trajectory=True,
        train_inverse_operator=True,
        uses_official_inverse_observation_operator=False,
    )
    assert vivid_style.support_status == "adapted"
    assert vivid_style.paper_table_eligible is False

    vivid_official = resolve_capability(
        "vivid",
        "shallow_water",
        "sparse_solution",
        "time_varying",
        "time_varying",
        load_full_trajectory=True,
        train_inverse_operator=True,
        uses_official_inverse_observation_operator=True,
    )
    assert vivid_official.support_status == "official_adapter"
    assert vivid_official.paper_table_eligible is True


def test_burger_time_varying_da_capability_requires_full_trajectory():
    var4d = resolve_capability(
        "var4d",
        "burger",
        "sparse_solution",
        "time_varying",
        "time_varying",
        load_full_trajectory=True,
    )
    assert var4d.support_status == "native"
    assert var4d.paper_table_eligible is True
