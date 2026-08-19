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
