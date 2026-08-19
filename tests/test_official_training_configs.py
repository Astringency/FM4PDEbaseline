from __future__ import annotations

from pathlib import Path

import yaml

from baselines.run import _method_budget_fields


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
    assert methods["senseiver"]["early_stopping_patience"] == 100
    assert methods["voronoicnn"]["width"] == 48
    assert methods["voronoicnn"]["early_stopping_patience"] == 100


def test_ifno_budget_records_all_three_stages_and_total():
    budget = _method_budget_fields(
        {"ifno_pretrain_epochs": 200, "vae_pretrain_epochs": 100, "joint_epochs": 200},
        "ifno",
    )

    assert budget["total_training_epochs"] == 500
    assert budget["method_budget_label"] == (
        "ifno_pretrain_epochs=200,vae_pretrain_epochs=100,joint_epochs=200,total_epochs=500"
    )


def test_central_paper_config_is_the_only_method_recipe_source():
    standalone_dir = ROOT / "baselines" / "configs" / "paper"
    assert not standalone_dir.exists() or not list(standalone_dir.glob("*.yaml"))


def test_baseline_summary_does_not_reference_removed_method_recipe_files():
    text = (ROOT / "docs" / "RecFNO_Senseiver_VoronoiCNN_baselines.md").read_text(encoding="utf-8")

    assert "baselines/configs/paper/recfno.yaml" not in text
    assert "baselines/configs/paper/senseiver.yaml" not in text
    assert "baselines/configs/paper/voronoicnn.yaml" not in text
