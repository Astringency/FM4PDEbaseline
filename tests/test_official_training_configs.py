from __future__ import annotations

from pathlib import Path

import yaml

from baselines.run import _method_budget_fields, build_method_config, parse_args


ROOT = Path(__file__).resolve().parents[1]


def test_supervised_baseline_configs_preserve_upstream_training_recipes():
    config = yaml.safe_load((ROOT / "baselines" / "configs" / "paper.yaml").read_text(encoding="utf-8"))
    methods = config["method_by_baseline"]

    assert {key: methods["fno"][key] for key in ("optimizer", "training_loss", "lr_scheduler")} == {
        "optimizer": "adamw",
        "training_loss": "h1",
        "lr_scheduler": "step",
    }
    assert {key: methods["recfno"][key] for key in ("optimizer", "training_loss", "lr_scheduler")} == {
        "optimizer": "adam",
        "training_loss": "l1",
        "lr_scheduler": "exponential",
    }
    assert methods["senseiver"]["training_loss"] == "sum_mse"
    assert methods["senseiver"]["batch_pixels"] == 2048
    assert methods["senseiver"]["scheduler_monitor"] == "train_loss"
    assert methods["senseiver"]["early_stopping_patience"] == 20
    assert methods["voronoicnn"]["width"] == 48
    assert methods["voronoicnn"]["early_stopping_patience"] == 20


def test_ifno_budget_records_all_three_stages_and_total():
    budget = _method_budget_fields(
        {"ifno_pretrain_epochs": 200, "vae_pretrain_epochs": 100, "joint_epochs": 200},
        "ifno",
    )

    assert budget["total_training_epochs"] == 500
    assert budget["method_budget_label"] == (
        "ifno_pretrain_epochs=200,vae_pretrain_epochs=100,joint_epochs=200,total_epochs=500"
    )


def test_pinn_budget_records_adam_lbfgs_and_total_iterations():
    budget = _method_budget_fields(
        {
            "deepxde_native": True,
            "steps": 1000,
            "adam_iterations": 1000,
            "lbfgs_steps": 500,
        },
        "pinn_sparse",
    )

    assert budget["steps"] == 1000
    assert budget["adam_iterations"] == 1000
    assert budget["lbfgs_steps"] == 500
    assert budget["total_optimization_steps"] == 1500
    assert budget["method_budget_label"] == (
        "adam_iterations=1000,lbfgs_steps=500,total_optimization_steps=1500"
    )


def test_pinn_matrix_steps_override_controls_the_adam_phase():
    config = yaml.safe_load((ROOT / "baselines" / "configs" / "paper.yaml").read_text(encoding="utf-8"))
    args = parse_args(["--baseline", "pinn_sparse", "--pde", "poisson", "--steps", "250"])

    method = build_method_config(config, args)

    assert method["steps"] == 250
    assert method["adam_iterations"] == 250
    assert method["lbfgs_steps"] == 500


def test_pinn_dry_run_bounds_both_optimizer_phases():
    config = yaml.safe_load((ROOT / "baselines" / "configs" / "paper.yaml").read_text(encoding="utf-8"))
    args = parse_args(["--baseline", "pinn_sparse", "--pde", "poisson", "--dry-run"])

    method = build_method_config(config, args)

    assert method["adam_iterations"] == 1
    assert method["lbfgs_steps"] == 1


def test_central_paper_config_is_the_only_method_recipe_source():
    standalone_dir = ROOT / "baselines" / "configs" / "paper"
    assert not standalone_dir.exists() or not list(standalone_dir.glob("*.yaml"))


def test_baseline_summary_does_not_reference_removed_method_recipe_files():
    text = (ROOT / "docs" / "RecFNO_Senseiver_VoronoiCNN_baselines.md").read_text(encoding="utf-8")

    assert "baselines/configs/paper/recfno.yaml" not in text
    assert "baselines/configs/paper/senseiver.yaml" not in text
    assert "baselines/configs/paper/voronoicnn.yaml" not in text
