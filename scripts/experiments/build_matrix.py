#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.experiment_matrix import (
    ALL_BASELINES,
    ALL_PDES,
    FUTURE_PDES,
    PER_INSTANCE_BASELINES,
    TIME_VARYING_SENSOR_BASELINES,
    capability_skip_row,
    resolve_capability,
)
from baselines.configuration import resolve_method_config
from baselines.common.data_files import files_for_pde, load_data_files_from_config
from scripts.experiments.provenance import (
    DEFAULT_SENSOR_PROTOCOL_VERSION,
    DEFAULT_TASK_PROTOCOL_VERSION,
    FINGERPRINT_FIELDS,
    HISTORICAL_EXPERIMENT_NAMESPACE,
    HistoricalExperimentError,
    MATRIX_SCHEMA_VERSION,
    SUMMARY_SCHEMA_VERSION,
    baseline_code_sha256,
    baseline_config_sha256,
    canonical_sha256,
    data_content_sha256,
    repository_revision,
    reject_historical_experiment_path,
    run_fingerprint,
    sha256_file,
    summary_validation_reasons,
    validate_full_data_manifest,
)


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def progress(message: str) -> None:
    print(f"[{timestamp()}] {message}", file=sys.stderr, flush=True)


GROUP_TO_TASK = {
    "full_forward_main": "forward",
    "full_inverse_main": "inverse",
    "sparse_solution_main_amortized": "sparse_solution",
    "sparse_solution_burger_time_slices": "sparse_solution",
    "time_varying_da_burger_time_slices": "sparse_solution",
    "sparse_solution_main_physics": "sparse_solution",
    "sparse_forward_main_amortized": "sparse_forward",
    "sparse_forward_main_physics": "sparse_forward",
    "sparse_inverse_main_amortized": "sparse_inverse",
    "sparse_inverse_main": "sparse_inverse",
    "time_varying_da_main": "sparse_solution",
    "sensor_count_ablation": "sparse_solution",
    "noise_ablation": "sparse_solution",
    "sensor_mode_ablation": "sparse_solution",
    "time_varying_sensor_ablation": "sparse_solution",
    "runtime_budget_ablation": "sparse_solution",
    "runtime_budget_sparse_forward": "sparse_forward",
    "runtime_budget_pcbnn": "sparse_solution",
    "time_varying_da_runtime_budget": "sparse_solution",
    "train_size_ablation": "sparse_solution",
    "sparse_solution_multicondition_train": "sparse_solution_multicondition",
}

DEFAULT_BASELINES_BY_GROUP = {
    "full_forward_main": ["fno", "deeponet", "ifno"],
    "full_inverse_main": ["fno", "deeponet", "ifno"],
    "sparse_solution_main_amortized": ["recfno", "senseiver", "voronoicnn"],
    "sparse_solution_burger_time_slices": ["recfno", "senseiver", "voronoicnn"],
    "time_varying_da_burger_time_slices": ["var4d", "vivid"],
    "sparse_solution_main_physics": ["pinn_sparse", "pc_bnn", "pde_opt"],
    "sparse_forward_main_amortized": ["recfno", "senseiver", "voronoicnn"],
    "sparse_forward_main_physics": ["pinn_sparse", "pde_opt"],
    "sparse_inverse_main_amortized": ["recfno", "senseiver", "voronoicnn"],
    "sparse_inverse_main": ["pinn_sparse", "pde_opt"],
    "time_varying_da_main": ["var4d", "vivid"],
    "sensor_count_ablation": ["recfno", "senseiver", "voronoicnn"],
    "noise_ablation": ["recfno", "senseiver", "voronoicnn"],
    "sensor_mode_ablation": ["recfno", "senseiver", "voronoicnn"],
    "time_varying_sensor_ablation": ["var4d", "vivid", "senseiver"],
    "runtime_budget_ablation": ["pinn_sparse", "pc_bnn", "pde_opt", "var4d", "vivid"],
    "runtime_budget_sparse_forward": ["pinn_sparse", "pde_opt"],
    "runtime_budget_pcbnn": ["pc_bnn"],
    "time_varying_da_runtime_budget": ["var4d", "vivid"],
    "train_size_ablation": ["recfno", "senseiver", "voronoicnn"],
    "sparse_solution_multicondition_train": ["recfno", "senseiver", "voronoicnn"],
}

VALID_EXPERIMENT_KINDS = {"main", "ablation"}
VALID_COMPARISON_TRACKS = {
    "unified_adapted",
    "official_native",
    "sparse_solution_multicondition",
}
VALID_ABLATION_FACTORS = {
    "sensor_count",
    "noise_level",
    "sensor_mode",
    "time_varying_sensor_count",
    "runtime_budget",
    "train_size",
    "sparse_solution_multicondition",
}
FORMAL_DESIGN_OVERRIDE_ENV = (
    "SEEDS",
    "TRAIN_SIZE",
    "TRAIN_SIZES",
    "VAL_SIZE",
    "TEST_SIZE",
    "TRAIN_SHARDS",
    "SENSOR_COUNTS",
    "SENSOR_MODES",
    "NOISE_LEVELS",
    "SENSOR_SEED",
    "BATCH_SIZE",
    "EPOCHS",
    "DEVICE",
    "SCALAR_PARAM_MODE",
    "NUM_WORKERS",
    "PIN_MEMORY",
    "PERSISTENT_WORKERS",
    "PREFETCH_FACTOR",
    "PINN_STEPS",
    "PDEOPT_STEPS",
    "VAR4D_STEPS",
    "PCBNN_STEPS",
    "VIVID_REFINE_STEPS",
    "PCBNN_PARTICLES",
    "FULL_ABLATION_ALL",
)
FACTOR_TO_VARIED_FIELDS = {
    "sensor_count": {"num_sensors"},
    "noise_level": {"noise_level"},
    "sensor_mode": {"sensor_mode"},
    "time_varying_sensor_count": {"num_sensors"},
    "runtime_budget": {"steps", "refine_steps", "particles"},
    "train_size": {"train_size"},
    "sparse_solution_multicondition": set(),
}

# Public alias used by downstream tooling. The fingerprint list is defined in
# one place alongside the summary validation contract.
HASH_FIELDS = list(FINGERPRINT_FIELDS)
DEPENDENCY_PENDING_SHA256 = "0" * 64
RESUME_IDENTITY_FIELDS = tuple(
    field for field in FINGERPRINT_FIELDS if field != "baseline_code_sha256"
)
# Compatibility bridge for completed cohorts across the inverse-checkpoint and
# evaluation-resume migrations. These changes affect orchestration/evaluation,
# not fitted model state. Limiting the exception to exact before/after digests
# ensures future training-code changes still invalidate affected rows normally.
INVERSE_REUSE_MIGRATION_CODE_SHA256 = {
    "4edf9ce579e62f0a8afd7616d064b272086daeb950c5e7590202a70940c7ca10":
        "8aecf00b56eb01dc0924fea0478350ff064d076116be726662e55f27087411aa",
    "e4185f35320beeb0f210d1e9ae9301d04ef1510c775e34d1c1276fa94a112a9b":
        "8aecf00b56eb01dc0924fea0478350ff064d076116be726662e55f27087411aa",
    "4a568785cd8658fab71db0845e6a3f2d9bb643475dc07e617ec2b9b733610e8e":
        "b68c154026fa03b73c9bb213207fb0a2c375daeda8439bfdb2be47a25ca78e58",
    "4e66c4d070f282059838579a460cd16963590fd704559851b21871618bafe1b7":
        "b68c154026fa03b73c9bb213207fb0a2c375daeda8439bfdb2be47a25ca78e58",
    "ee2c58e7d2737f6d3b520248c90a03164f1c4ab0fbc414a38a00936beaf5bcbf":
        "df279b550811cbaaead3cb4ec732bbcca38413d24aeb03a6c6a92628fdaaf01f",
    "e7ab63c7b3a26c7b2e3e20afc102f1faaad8235c27e0bde088081418bea5d525":
        "6c2c1a769c6c4d8da4bbd4bbd4d0494cc931c73aff77f9b58158e74662081fe4",
    "3d8b8127c99eb40e043383e13e588f45c10396f8ae22a34991b18d063f8e210c":
        "c1fbde9489c8e5010217bf553243dba10cc63b014118b2ed3d441ac3a0328541",
    "514375a8d32e97c7df542d82d05cd1cc9dd1de48170cd7c87807a7e93e53e30b":
        "c1fbde9489c8e5010217bf553243dba10cc63b014118b2ed3d441ac3a0328541",
    "7866332069fe38fee9ca87fceca82762d1d4434de1bf633355f3456fe2e84226":
        "9da5d1e68cf462ce253e5ab31878166846d2325c7297c8b2c5e339c81ea7c4f9",
    "a60d0df14c54a1786078c8b7e65a4bf8965cf55f6b3da3f4e5212a980c98a02d":
        "9da5d1e68cf462ce253e5ab31878166846d2325c7297c8b2c5e339c81ea7c4f9",
    "9e8599e914d59497aab854245b65ef275532bd9f2f0217495c7bc6b20e5d97f4":
        "c14dea4277e6bf38a52c64ab70de3a2bc040d481d38b88d8241fa5200b7c7743",
    "e1d5f4fb8bfacb72c8964597e5870f3e2ec96e365a39b88b1343173b385da482":
        "c14dea4277e6bf38a52c64ab70de3a2bc040d481d38b88d8241fa5200b7c7743",
    "fefa36f8c90cec82103dd57a4f1afb7be7783cb55df3b0efc924da47556d75ef":
        "d7404e8d8e83d78b438dc8c61d8f66e48a0c06ec736cb6512bd0a68496f68767",
    "abe71061bb24048e5649bfcc8fa048e5b236792c4b550ff9315dbb5c5f8ed752":
        "d7404e8d8e83d78b438dc8c61d8f66e48a0c06ec736cb6512bd0a68496f68767",
    "54c8dbfe34973b008f7bfdf27e12acfd32a619a5a785528438177cfb6e07cb4b":
        "b67878c254cb8cb42eb7a1fc439c9e66d410303c0e2527261b34960bdb11244a",
    "5d828173a3e30fcf8d98b2000d5e6e2e0c1889431c0a6d5e4bb8866afe49e5f4":
        "b67878c254cb8cb42eb7a1fc439c9e66d410303c0e2527261b34960bdb11244a",
    "e53691458b57cbaa14d656769c59d43118a8a73ad833a333c7db1a147d138055":
        "f4485388d8962eb4422ad0fa3bfbdbb882c9f09898b41850cd46cb300acad0d2",
    "488a9f5753b3751eda2cc53e3d5978c9ac8479f304035a7a5f83296dd23d64f2":
        "f4485388d8962eb4422ad0fa3bfbdbb882c9f09898b41850cd46cb300acad0d2",
    "2cf7aca50f94bac68e5603c8205eb3e09b205c5dcb661ed1ce0595379315ca1c":
        "e8a1197a8baa166f72d4482ec18b557ddbe52a981cc8ee3c971ae7bccb9042a1",
    "4849bcda18c84356bff6812516ada404c593c7542010d71faffd051d5f7b6825":
        "e8a1197a8baa166f72d4482ec18b557ddbe52a981cc8ee3c971ae7bccb9042a1",
    "5da54a3181116a7f6834e262d7d1ff2630e0bd18e82b6ae22489feeb80885aff":
        "260daa268f8d8ddab72d15eb7814e12514838de83e7a380ada890702316e481c",
    "88252df6d11dc6b0538f6671c31838b19f8db0855a0b9114f3379327c070952b":
        "260daa268f8d8ddab72d15eb7814e12514838de83e7a380ada890702316e481c",
}

MATRIX_FIELDS = [
    "matrix_schema_version",
    "summary_schema_version",
    "run_id",
    "run_fingerprint",
    "run_name",
    "execution_mode",
    "source_train_run_id",
    "source_train_run_fingerprint",
    "source_train_task",
    "source_train_baseline_code_sha256",
    "comparison_track",
    "experiment_kind",
    "ablation_factor",
    "task_group",
    "task",
    "task_protocol_version",
    "sensor_protocol_version",
    "data_manifest_sha256",
    "data_manifest_path",
    "capability_status",
    "implementation_required",
    "paper_table_eligible",
    "unified_comparison_eligible",
    "official_native_eligible",
    "pde",
    "baseline",
    "seed",
    "sensor_seed",
    "source_train_seed",
    "train_size",
    "val_size",
    "test_size",
    "train_shards",
    "data_files",
    "num_sensors",
    "sensor_mode",
    "sensor_budget_mode",
    "noise_level",
    "condition_mode",
    "condition_probabilities",
    "evaluation_condition_modes",
    "train_only",
    "scalar_param_mode",
    "physics_metric_mode",
    "data_loading_mode",
    "num_workers",
    "pin_memory",
    "persistent_workers",
    "prefetch_factor",
    "load_full_trajectory",
    "batch_size",
    "epochs",
    "steps",
    "refine_steps",
    "particles",
    "device",
    "commit_hash",
    "config",
    "config_content_sha256",
    "baseline_config_sha256",
    "baseline_code_sha256",
    "experiment_config_sha256",
    "data_content_sha256",
    "checkpoint_path",
    "checkpoint_sha256",
    "dependency_run_id",
    "dependency_pending",
    "dependency_reason",
    "output_dir",
    "log_dir",
    "status_file",
    "skip_reason",
]

SUMMARY_DESIGN_FIELDS = [
    "task_group",
    "task",
    "pde",
    "baseline",
    "seed",
    "train_size",
    "num_sensors",
    "sensor_mode",
    "sensor_budget_mode",
    "noise_level",
    "steps",
    "refine_steps",
    "particles",
    "load_full_trajectory",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Build large-scale FM4PDE external-baseline experiment matrices.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", default=os.environ.get("OUT_ROOT", "outputs/baselines_large"))
    parser.add_argument("--matrix-name", default="")
    parser.add_argument("--include-skipped", action="store_true", help="Also include unsupported rows in the matrix with skip_reason set.")
    parser.add_argument("--comparison-track", choices=sorted(VALID_COMPARISON_TRACKS), default="")
    parser.add_argument(
        "--data-manifest",
        default="",
        help="Full, passing data_protocol_report.json to bind into every formal experiment row.",
    )
    return parser.parse_args(argv)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_data_manifest_binding(
    path: str | Path,
    *,
    experiment_config_path: str | Path | None = None,
    expected_pdes: list[str] | None = None,
    expected_data_files_config_sha256: str = "",
) -> tuple[str, str]:
    """Validate a full data-protocol report and return absolute path + SHA-256."""
    expected_config_sha256 = (
        sha256_file(experiment_config_path, root=ROOT)
        if experiment_config_path is not None
        else ""
    )
    manifest, manifest_path, manifest_sha256 = validate_full_data_manifest(
        path,
        expected_experiment_config_sha256=expected_config_sha256,
        expected_pdes=expected_pdes,
        expected_verifier_sha256=sha256_file(ROOT / "scripts" / "verify_data_protocol.py"),
        verify_source_signatures=True,
    )
    observed_data_files_sha256 = str(
        manifest.get("data_files_config_sha256", "") or ""
    )
    if (
        expected_data_files_config_sha256
        and observed_data_files_sha256 != expected_data_files_config_sha256
    ):
        raise ValueError(
            "data manifest was generated from a different explicit data-file configuration: "
            f"manifest={observed_data_files_sha256}, "
            f"expected={expected_data_files_config_sha256}"
        )
    return manifest_path, manifest_sha256


def build_matrix(
    cfg: dict[str, Any],
    output_root: str | Path,
    matrix_name: str,
    include_skipped: bool = False,
    emit_progress: bool = False,
    data_manifest: str | Path | None = None,
    experiment_config_path: str | Path | None = None,
    resume_rows: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    output_root = Path(output_root)
    _reject_historical_matrix_target(output_root, matrix_name)
    _validate_output_root_revision_safety(output_root)
    experiment_kind = _experiment_kind(cfg)
    ablation_factor = _ablation_factor(cfg, experiment_kind)
    comparison_track = _comparison_track(cfg)
    allow_multi = _as_bool(cfg.get("allow_multi_factor_grid", False))
    task_groups = list(cfg.get("task_groups", []))
    if not task_groups:
        raise ValueError("config must define at least one task_group")

    rows: list[dict[str, Any]] = []
    skipped: dict[tuple[Any, ...], dict[str, Any]] = {}
    global_defaults = _global_defaults(cfg)
    commit_hash = repository_revision(ROOT)
    experiment_config_sha256 = (
        sha256_file(experiment_config_path, root=ROOT)
        if experiment_config_path is not None
        else ""
    )
    if data_manifest:
        _reject_formal_design_environment_overrides()
        if experiment_config_path is None:
            raise ValueError(
                "data_manifest requires experiment_config_path so the full report can be bound to the exact design YAML"
            )
        data_manifest_path, data_manifest_sha256 = load_data_manifest_binding(
            data_manifest,
            experiment_config_path=experiment_config_path,
            expected_pdes=_configured_manifest_pdes(cfg),
            expected_data_files_config_sha256=str(
                global_defaults.get("data_files_sha256", "")
            ),
        )
        data_manifest_payload = json.loads(
            Path(data_manifest_path).read_text(encoding="utf-8")
        )
        data_content_digest = data_content_sha256(data_manifest_payload)
    else:
        # Direct construction remains useful for capability/design inspection.
        # Such formal rows cannot be executed or published until rebuilt from
        # the CLI with a passing full manifest.
        data_manifest_path, data_manifest_sha256 = "", ""
        data_content_digest = canonical_sha256(
            {"verification": "not_bound_to_full_data_manifest"}
        )

    for task_group in task_groups:
        if task_group not in GROUP_TO_TASK:
            raise ValueError(f"Unknown task_group {task_group!r}")
        group_cfg = dict(cfg.get("task_group_overrides", {}).get(task_group, {}) or {})
        task = str(group_cfg.get("task", GROUP_TO_TASK[task_group]))
        _validate_group_design(
            cfg=cfg,
            group_cfg=group_cfg,
            defaults=global_defaults,
            experiment_kind=experiment_kind,
            ablation_factor=ablation_factor,
            task_group=task_group,
            task=task,
            allow_multi=allow_multi,
        )
        pdes = _resolve_pdes(cfg, group_cfg, experiment_kind, ablation_factor)
        baselines = _resolve_baselines(cfg, task_group, task, group_cfg, experiment_kind, ablation_factor)
        if task == "sparse_solution_multicondition":
            invalid_pdes = sorted(set(pdes) - {"poisson", "helmholtz", "darcy", "nsnonbounded"})
            if invalid_pdes:
                if "burger" in invalid_pdes:
                    raise ValueError(
                        "sparse_solution_multicondition does not support Burgers trajectory semantics"
                    )
                raise ValueError(
                    f"sparse_solution_multicondition contains unsupported PDEs: {invalid_pdes}"
                )
            invalid_baselines = sorted(set(baselines) - {"recfno", "senseiver", "voronoicnn"})
            if invalid_baselines:
                raise ValueError(
                    "sparse_solution_multicondition supports only recfno, senseiver, and voronoicnn; "
                    f"got {invalid_baselines}"
                )
        extra_skip_baselines = list(group_cfg.get("extra_skip_baselines", []) or [])
        candidate_baselines = list(baselines)
        seeds = _env_list("SEEDS", group_cfg.get("seeds", global_defaults["seeds"]), int)
        expansion = _group_expansion(cfg, group_cfg, global_defaults, task_group, task)
        if emit_progress:
            progress(
                f"[matrix group] task_group={task_group} task={task} pdes={len(pdes)} "
                f"baselines={_short_list(baselines)} sensor_modes={_short_list(expansion['sensor_modes'])} "
                f"sensor_counts={_short_list(expansion['sensor_counts'])} noise_levels={_short_list(expansion['noise_levels'])} "
                f"seeds={_short_list(seeds)} train_sizes={_short_list(expansion['train_sizes'])}"
            )

        for pde in pdes:
            for baseline in extra_skip_baselines:
                if baseline in baselines:
                    continue
                budget_variants = _budget_variants(cfg, group_cfg, baseline, ablation_factor, experiment_kind)
                for sensor_mode in expansion["sensor_modes"]:
                    skipped_row = {
                        "matrix_name": matrix_name,
                        "experiment_kind": experiment_kind,
                        "ablation_factor": ablation_factor,
                        "comparison_track": comparison_track,
                        "task_group": task_group,
                        "task": task,
                        "pde": pde,
                        "baseline": baseline,
                        "sensor_mode": sensor_mode,
                        "reason": "baseline is intentionally skipped for this task_group",
                        "would_have_expanded": (
                            len(seeds)
                            * len(expansion["sensor_counts"])
                            * len(expansion["noise_levels"])
                            * len(expansion["train_sizes"])
                            * len(budget_variants)
                        ),
                    }
                    _merge_skip(skipped, skipped_row)
                    if include_skipped:
                        rows.append(
                            _skipped_matrix_row(
                                skipped_row,
                                global_defaults,
                                output_root,
                                commit_hash,
                                experiment_config_sha256,
                                data_manifest_path,
                                data_manifest_sha256,
                                data_content_digest,
                            )
                        )
            for baseline in candidate_baselines:
                budget_variants = _budget_variants(cfg, group_cfg, baseline, ablation_factor, experiment_kind)
                for sensor_mode in expansion["sensor_modes"]:
                    capability = resolve_capability(
                        baseline,
                        pde,
                        task,
                        "" if sensor_mode == "none" else sensor_mode,
                        task_group,
                        load_full_trajectory=_matrix_load_full_trajectory(group_cfg, global_defaults, baseline),
                        train_inverse_operator=_matrix_train_inverse_operator(cfg, group_cfg, baseline),
                        uses_official_inverse_observation_operator=_matrix_uses_official_inverse_observation_operator(cfg, group_cfg, baseline),
                    )
                    reason = capability.reason if capability.support_status == "unsupported" else ""
                    if not reason and comparison_track == "unified_adapted" and not capability.unified_comparison_eligible:
                        reason = (
                            f"{capability.reason}; this debug adaptation is not eligible for the unified "
                            "comparison track"
                        )
                    if not reason and comparison_track == "official_native" and not capability.official_native_eligible:
                        reason = (
                            f"{capability.baseline}/{capability.task_family} is not eligible for the strict "
                            "official-native verification track"
                        )
                    if reason:
                        skipped_row = capability_skip_row(
                            capability,
                            task_group=task_group,
                            extra={
                                "matrix_name": matrix_name,
                                "experiment_kind": experiment_kind,
                                "ablation_factor": ablation_factor,
                                "comparison_track": comparison_track,
                                "official_native_eligible": capability.official_native_eligible,
                                "reason": reason,
                                "unsupported_reason": reason if capability.support_status == "unsupported" else "",
                                "would_have_expanded": (
                                    len(seeds)
                                    * len(expansion["sensor_counts"])
                                    * len(expansion["noise_levels"])
                                    * len(expansion["train_sizes"])
                                    * len(budget_variants)
                                ),
                            },
                        )
                        _merge_skip(skipped, skipped_row)
                        if include_skipped:
                            rows.append(
                                _skipped_matrix_row(
                                    skipped_row,
                                    global_defaults,
                                    output_root,
                                    commit_hash,
                                    experiment_config_sha256,
                                    data_manifest_path,
                                    data_manifest_sha256,
                                    data_content_digest,
                                )
                            )
                        continue
                    for seed in seeds:
                        for train_size in expansion["train_sizes"]:
                            for num_sensors in expansion["sensor_counts"]:
                                for noise_level in expansion["noise_levels"]:
                                    for budget in budget_variants:
                                        row = _make_run_row(
                                            cfg=cfg,
                                            group_cfg=group_cfg,
                                            defaults=global_defaults,
                                            output_root=output_root,
                                            matrix_name=matrix_name,
                                            experiment_kind=experiment_kind,
                                            ablation_factor=ablation_factor,
                                            comparison_track=comparison_track,
                                            capability=capability,
                                            task_group=task_group,
                                            task=task,
                                            pde=str(pde),
                                            baseline=str(baseline),
                                            seed=int(seed),
                                            train_size=int(train_size),
                                            num_sensors=int(num_sensors),
                                            sensor_mode=str(sensor_mode),
                                            noise_level=float(noise_level),
                                            budget=budget,
                                            commit_hash=commit_hash,
                                            experiment_config_sha256=experiment_config_sha256,
                                            data_manifest_path=data_manifest_path,
                                            data_manifest_sha256=data_manifest_sha256,
                                            data_content_digest=data_content_digest,
                                        )
                                        rows.append(row)

    if any(row.get("task") == "sparse_solution_multicondition" for row in rows):
        rows = _expand_multicondition_evaluation_rows(
            rows, cfg, output_root, matrix_name
        )
    if resume_rows:
        rows = _preserve_completed_rows(rows, resume_rows)
    rows = _bind_eval_only_dependencies(rows, cfg, output_root, matrix_name)
    _ensure_unique_run_ids(rows)
    skipped_rows = list(skipped.values())
    summary = _summary(
        rows,
        skipped_rows,
        matrix_name,
        experiment_kind,
        ablation_factor,
        comparison_track,
        experiment_config_sha256,
        data_manifest_path,
        data_manifest_sha256,
        data_content_digest,
    )
    return rows, skipped_rows, summary


def _expand_multicondition_evaluation_rows(
    rows: list[dict[str, Any]],
    cfg: dict[str, Any],
    output_root: Path,
    matrix_name: str,
) -> list[dict[str, Any]]:
    modes = [str(value) for value in cfg.get("evaluation_condition_modes", ["a_only", "u_only", "both"])]
    if modes != ["a_only", "u_only", "both"]:
        raise ValueError(
            "sparse_solution_multicondition evaluation_condition_modes must be exactly "
            "[a_only, u_only, both]"
        )
    expanded = list(rows)
    for source in rows:
        if source.get("task") != "sparse_solution_multicondition":
            continue
        if source.get("execution_mode") != "train":
            raise ValueError("multicondition source rows must be training rows")
        for condition_mode in modes:
            row = dict(source)
            row.update(
                {
                    "task_group": f"sparse_solution_multicondition_eval_{condition_mode}",
                    "execution_mode": "eval_only",
                    "condition_mode": condition_mode,
                    "train_only": False,
                    "source_train_run_id": source["run_id"],
                    "source_train_run_fingerprint": source["run_fingerprint"],
                    "source_train_task": source["task"],
                    "source_train_baseline_code_sha256": source["baseline_code_sha256"],
                    "source_train_seed": source["seed"],
                    "checkpoint_path": str(
                        Path(str(source["output_dir"])) / f"{source['run_id']}.pt"
                    ),
                    "checkpoint_sha256": DEPENDENCY_PENDING_SHA256,
                    "dependency_run_id": source["run_id"],
                    "dependency_pending": True,
                    "dependency_reason": "source training checkpoint is not available",
                }
            )
            row["run_fingerprint"] = run_fingerprint(row)
            row["run_id"] = _run_id(row)
            row["run_name"] = _run_name(row)
            row["output_dir"] = str(_run_output_dir(output_root, matrix_name, row))
            row["log_dir"] = str(_run_log_dir(output_root, matrix_name, row))
            row["status_file"] = str(Path(row["output_dir"]) / "run.status.json")
            expanded.append(row)
    return expanded


def _validate_condition_probabilities_config(value: Any) -> dict[str, float]:
    if value is None:
        value = {
            "a_only": 0.3333333333,
            "u_only": 0.3333333333,
            "both": 0.3333333334,
        }
    if not isinstance(value, dict) or set(value) != {"a_only", "u_only", "both"}:
        raise ValueError(
            "condition_probabilities must define exactly a_only, u_only, and both"
        )
    probabilities = {key: float(value[key]) for key in ("a_only", "u_only", "both")}
    if any(probability < 0.0 for probability in probabilities.values()):
        raise ValueError("condition_probabilities must all be non-negative")
    total = sum(probabilities.values())
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"condition_probabilities must sum to 1, got {total:.12g}")
    return probabilities


def _preserve_completed_rows(
    generated_rows: list[dict[str, Any]],
    previous_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep completed semantic runs stable across orchestration-only code changes.

    Baseline code remains part of a new run's identity.  A previously completed
    row, however, is immutable evidence and should remain addressable when only
    the surrounding orchestration/evaluation code changes.  Match every design
    and data field except the code digest, then require the old summary and
    checkpoint to pass their original provenance contract before preserving it.
    """

    previous_by_identity: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in previous_rows:
        previous_by_identity.setdefault(_resume_identity(row), []).append(row)
    preserved: list[dict[str, Any]] = []
    for row in generated_rows:
        replacement = None
        for candidate in previous_by_identity.get(_resume_identity(row), []):
            candidate = dict(candidate)
            previous_code = str(candidate.get("baseline_code_sha256", ""))
            current_code = str(row.get("baseline_code_sha256", ""))
            if (
                previous_code != current_code
                and INVERSE_REUSE_MIGRATION_CODE_SHA256.get(previous_code)
                != current_code
            ):
                continue
            # The verifier report is regenerated on every workflow start and
            # contains a timestamp, while data_content_sha256 is the stable
            # identity already matched above.  Refresh only the audit binding;
            # keep the completed run/checkpoint identity unchanged.
            for field in (
                "data_manifest_sha256",
                "data_manifest_path",
                "experiment_config_sha256",
            ):
                candidate[field] = row.get(field, candidate.get(field))
            summary_path = Path(str(candidate.get("output_dir", ""))) / "summary.json"
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(summary, dict) and not summary_validation_reasons(candidate, summary):
                replacement = dict(candidate)
                break
        preserved.append(replacement or row)
    return preserved


def _resume_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    values: list[Any] = []
    for field in RESUME_IDENTITY_FIELDS:
        value = row.get(field)
        if isinstance(value, (dict, list)):
            value = json.dumps(value, sort_keys=True, separators=(",", ":"))
        values.append(value)
    if "data_files" in row:
        values.append(
            json.dumps(row.get("data_files"), sort_keys=True, separators=(",", ":"))
        )
    return tuple(values)


def _bind_eval_only_dependencies(
    rows: list[dict[str, Any]],
    cfg: dict[str, Any],
    output_root: Path,
    matrix_name: str,
) -> list[dict[str, Any]]:
    """Bind eval-only rows to completed source checkpoints when available."""

    group_configs = dict(cfg.get("task_group_overrides", {}) or {})
    bound: list[dict[str, Any]] = []
    for original in rows:
        if original.get("execution_mode") != "eval_only":
            bound.append(original)
            continue
        row = dict(original)
        group_cfg = dict(group_configs.get(str(row["task_group"]), {}) or {})
        reuse_ifno_forward = (
            row.get("baseline") == "ifno"
            and row.get("task_group") == "full_inverse_main"
            and row.get("task") == "inverse"
        )
        source_group = str(
            group_cfg.get(
                "source_task_group",
                (
                    "full_forward_main"
                    if reuse_ifno_forward
                    else "sparse_solution_multicondition_train"
                    if row.get("task") == "sparse_solution_multicondition"
                    else ""
                ),
            )
        )
        source_task = str(
            group_cfg.get(
                "source_task",
                "forward"
                if reuse_ifno_forward
                else "sparse_solution_multicondition"
                if row.get("task") == "sparse_solution_multicondition"
                else "",
            )
        )
        if not source_group or not source_task:
            raise ValueError(
                f"eval-only task group {row['task_group']!r} must define "
                "source_task_group and source_task"
            )
        candidates = [
            candidate
            for candidate in rows
            if candidate.get("execution_mode") == "train"
            and candidate.get("task_group") == source_group
            and candidate.get("task") == source_task
            and candidate.get("baseline") == row.get("baseline")
            and candidate.get("pde") == row.get("pde")
            and candidate.get("seed") == row.get("seed")
            and candidate.get("train_size") == row.get("train_size")
            and candidate.get("val_size") == row.get("val_size")
            and candidate.get("train_shards") == row.get("train_shards")
            and candidate.get("batch_size") == row.get("batch_size")
            and candidate.get("epochs") == row.get("epochs")
            and candidate.get("baseline_config_sha256")
            == row.get("baseline_config_sha256")
            and candidate.get("data_content_sha256") == row.get("data_content_sha256")
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"eval-only row {row['task_group']}/{row['baseline']}/{row['pde']}/seed={row['seed']} "
                f"requires exactly one source row in {source_group!r}, found {len(candidates)}"
            )
        source = candidates[0]
        checkpoint_path, checkpoint_sha256, reason = _validated_source_checkpoint(source)
        if not checkpoint_path:
            checkpoint_path = str(
                Path(str(source["output_dir"])) / f"{source['run_id']}.pt"
            )
            checkpoint_sha256 = DEPENDENCY_PENDING_SHA256
        row.update(
            {
                "source_train_run_id": source["run_id"],
                "source_train_run_fingerprint": source["run_fingerprint"],
                "source_train_task": source["task"],
                "source_train_baseline_code_sha256": source[
                    "baseline_code_sha256"
                ],
                "source_train_seed": source["seed"],
                "checkpoint_path": checkpoint_path,
                "checkpoint_sha256": checkpoint_sha256,
                "dependency_run_id": source["run_id"],
                "dependency_pending": bool(reason),
                "dependency_reason": reason,
            }
        )
        row["run_fingerprint"] = run_fingerprint(row)
        row["run_id"] = _run_id(row)
        row["run_name"] = _run_name(row)
        row["output_dir"] = str(_run_output_dir(output_root, matrix_name, row))
        row["log_dir"] = str(_run_log_dir(output_root, matrix_name, row))
        row["status_file"] = str(Path(row["output_dir"]) / "run.status.json")
        bound.append(row)
    return bound


def _validated_source_checkpoint(
    source: dict[str, Any],
) -> tuple[str, str, str]:
    summary_path = Path(str(source.get("output_dir", ""))) / "summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return "", "", "source summary is not available"
    if not isinstance(summary, dict):
        return "", "", "source summary is not an object"
    reasons = summary_validation_reasons(source, summary)
    if reasons:
        return "", "", "source run is incomplete: " + ",".join(reasons)
    checkpoint_path = str(summary.get("checkpoint_path", "") or "")
    checkpoint_sha256 = str(summary.get("checkpoint_sha256", "") or "")
    if not checkpoint_path or not checkpoint_sha256:
        return "", "", "source summary has no checkpoint"
    return checkpoint_path, checkpoint_sha256, ""


def _validate_output_root_revision_safety(output_root: str | Path) -> None:
    """Reject in-repository outputs that would invalidate their own revision."""
    resolved_root = Path(output_root).expanduser().resolve()
    repository_root = ROOT.resolve()
    try:
        relative = resolved_root.relative_to(repository_root)
    except ValueError:
        return
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", "--", str(relative)],
        cwd=repository_root,
        check=False,
    )
    if ignored.returncode != 0:
        raise ValueError(
            "output_root is inside the repository but is not git-ignored; writing the matrix "
            "would change commit_hash immediately. Use an ignored outputs/ path or an external directory: "
            f"{resolved_root}"
        )


def _reject_historical_matrix_target(
    output_root: str | Path, matrix_name: str
) -> None:
    """Protect the original v2 matrix namespace before any build or write."""
    reject_historical_experiment_path(output_root, field="matrix output_root")
    if (
        str(matrix_name) in {"", ".", ".."}
        or re.fullmatch(r"[A-Za-z0-9_.-]+", str(matrix_name)) is None
    ):
        raise ValueError(
            "matrix_name must be a safe basename containing only letters, digits, '.', '_', or '-': "
            f"{matrix_name!r}"
        )
    if str(matrix_name) == HISTORICAL_EXPERIMENT_NAMESPACE:
        raise HistoricalExperimentError(
            f"refusing to rebuild historical matrix {matrix_name!r}; "
            "use main_results and preserve the original evidence"
        )


def _reject_formal_design_environment_overrides() -> None:
    """Formal matrices must describe the reviewed YAML, not shell state."""
    active = {
        name: os.environ[name]
        for name in FORMAL_DESIGN_OVERRIDE_ENV
        if os.environ.get(name) not in {None, ""}
    }
    if active:
        raise ValueError(
            "formal matrix generation forbids design environment overrides; "
            f"update the experiment YAML instead: {sorted(active)}"
        )


def _configured_manifest_pdes(cfg: dict[str, Any]) -> list[str]:
    pdes: list[str] = []

    def add(values: Any) -> None:
        if values is None or (isinstance(values, str) and values.lower() == "all"):
            return
        for value in ([values] if isinstance(values, str) else values):
            name = str(value)
            if name not in pdes:
                pdes.append(name)

    add(cfg.get("pdes"))
    for group in (cfg.get("task_group_overrides", {}) or {}).values():
        if isinstance(group, dict):
            add(group.get("pdes"))
    if not pdes:
        raise ValueError("formal experiment config must declare an explicit PDE cohort")
    if _is_multicondition_config(cfg):
        return _restrict_multicondition_pdes(pdes)
    return pdes


def _is_multicondition_config(cfg: dict[str, Any]) -> bool:
    return bool(
        str(cfg.get("task", "")) == "sparse_solution_multicondition"
        or "sparse_solution_multicondition_train" in cfg.get("task_groups", [])
    )


def _restrict_multicondition_pdes(configured: list[str]) -> list[str]:
    """Apply the launcher's optional PDE cohort selector to this ablation only."""
    raw = os.environ.get("PDE_LIST", "").strip()
    if not raw:
        return list(configured)
    requested = [value for value in re.split(r"[,\s]+", raw) if value]
    if len(requested) != len(set(requested)):
        raise ValueError(f"PDE_LIST contains duplicate entries: {requested}")
    if "burger" in requested:
        raise ValueError(
            "sparse_solution_multicondition does not support Burgers trajectory semantics"
        )
    supported = {"poisson", "helmholtz", "darcy", "nsnonbounded"}
    unsupported = sorted(set(requested) - supported)
    if unsupported:
        raise ValueError(
            "PDE_LIST contains PDEs unsupported by sparse_solution_multicondition: "
            f"{unsupported}"
        )
    unavailable = sorted(set(requested) - set(configured))
    if unavailable:
        raise ValueError(
            f"PDE_LIST selects PDEs not declared by the experiment config: {unavailable}"
        )
    return requested


def write_outputs(rows: list[dict[str, Any]], skipped_rows: list[dict[str, Any]], summary: dict[str, Any], output_root: str | Path, matrix_name: str) -> None:
    output_root = Path(output_root)
    _reject_historical_matrix_target(output_root, matrix_name)
    matrix_dir = output_root / "matrices"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = matrix_dir / f"{matrix_name}.jsonl"
    tsv_path = matrix_dir / f"{matrix_name}.tsv"
    summary_path = matrix_dir / f"{matrix_name}_summary.json"
    skipped_matrix_path = matrix_dir / f"{matrix_name}_skipped.jsonl"
    skipped_global_path = output_root / "skipped_combinations.jsonl"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    _write_tsv(tsv_path, rows)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    with skipped_matrix_path.open("w", encoding="utf-8") as f:
        for row in skipped_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    existing_global = _read_jsonl(skipped_global_path)
    preserved_global = [row for row in existing_global if str(row.get("matrix_name", "")) != matrix_name]
    skipped_global_path.parent.mkdir(parents=True, exist_ok=True)
    with skipped_global_path.open("w", encoding="utf-8") as f:
        for row in preserved_global + skipped_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _experiment_kind(cfg: dict[str, Any]) -> str:
    kind = str(cfg.get("experiment_kind", "") or "")
    if kind not in VALID_EXPERIMENT_KINDS:
        raise ValueError(f"experiment_kind must be one of {sorted(VALID_EXPERIMENT_KINDS)}, got {kind!r}")
    return kind


def _ablation_factor(cfg: dict[str, Any], experiment_kind: str) -> str:
    factor = str(cfg.get("ablation_factor", "") or "")
    if experiment_kind == "main":
        if factor:
            raise ValueError("main experiments must not set ablation_factor")
        return ""
    if factor not in VALID_ABLATION_FACTORS:
        raise ValueError(f"ablation_factor must be one of {sorted(VALID_ABLATION_FACTORS)}, got {factor!r}")
    return factor


def _comparison_track(cfg: dict[str, Any]) -> str:
    track = str(cfg.get("comparison_track", "unified_adapted") or "unified_adapted")
    if track not in VALID_COMPARISON_TRACKS:
        raise ValueError(f"comparison_track must be one of {sorted(VALID_COMPARISON_TRACKS)}, got {track!r}")
    return track


def _global_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    train_size = _env_int("TRAIN_SIZE", int(cfg.get("train_size", 50000)))
    data_files, data_files_config_path, data_files_sha256 = load_data_files_from_config(
        cfg, repository_root=ROOT
    )
    return {
        "seeds": _env_list("SEEDS", cfg.get("seeds", [1]), int),
        "train_size": train_size,
        "val_size": _env_int("VAL_SIZE", int(cfg.get("val_size", 5000))),
        "test_size": _env_int("TEST_SIZE", int(cfg.get("test_size", 1000))),
        "train_shards": _env_int("TRAIN_SHARDS", int(cfg.get("train_shards", 5))),
        "data_files": data_files,
        "data_files_config_path": data_files_config_path,
        "data_files_sha256": data_files_sha256,
        "sensor_counts": _env_list("SENSOR_COUNTS", _first_present(cfg, ["sensor_counts", "num_sensors", "sensor_count"], [500]), int),
        "sensor_modes": _env_list(
            "SENSOR_MODES",
            _first_present(cfg, ["sensor_modes", "sensor_mode"], ["random_per_sample"]),
            str,
        ),
        "noise_levels": _env_list("NOISE_LEVELS", _first_present(cfg, ["noise_levels", "noise_level"], [0.0]), float),
        "train_sizes": _env_list("TRAIN_SIZES", cfg.get("train_sizes", [train_size]), int),
        "data_loading_mode": str(cfg.get("data_loading_mode", "eager")),
        "num_workers": _env_int("NUM_WORKERS", int(cfg.get("num_workers", 4))),
        "pin_memory": _env_bool("PIN_MEMORY", bool(cfg.get("pin_memory", True))),
        "persistent_workers": _env_bool("PERSISTENT_WORKERS", bool(cfg.get("persistent_workers", True))),
        "prefetch_factor": _env_int("PREFETCH_FACTOR", int(cfg.get("prefetch_factor", 2))),
        "device": os.environ.get("DEVICE", str(cfg.get("device", "cuda"))),
        "config": str(cfg.get("config", "baselines/configs/paper.yaml")),
        "scalar_param_mode": os.environ.get("SCALAR_PARAM_MODE", ""),
        "load_full_trajectory": bool(cfg.get("load_full_trajectory", False)),
    }


def _matrix_load_full_trajectory(group_cfg: dict[str, Any], defaults: dict[str, Any], baseline: str) -> bool:
    resources = group_cfg.get("resources", {}) if isinstance(group_cfg.get("resources", {}), dict) else {}
    baseline_resources = resources.get(baseline, {}) if isinstance(resources.get(baseline, {}), dict) else {}
    return bool(group_cfg.get("load_full_trajectory", baseline_resources.get("load_full_trajectory", defaults.get("load_full_trajectory", False))))


def _matrix_train_inverse_operator(cfg: dict[str, Any], group_cfg: dict[str, Any], baseline: str) -> bool:
    resources = dict(cfg.get("resources", {}) or {})
    baseline_resources = dict(resources.get(baseline, {}) or {})
    group_resources = dict(group_cfg.get("resources", {}) or {})
    group_baseline_resources = dict(group_resources.get(baseline, {}) or {})
    return bool(group_cfg.get("train_inverse_operator", group_baseline_resources.get("train_inverse_operator", baseline_resources.get("train_inverse_operator", False))))


def _matrix_uses_official_inverse_observation_operator(cfg: dict[str, Any], group_cfg: dict[str, Any], baseline: str) -> bool:
    resources = dict(cfg.get("resources", {}) or {})
    baseline_resources = dict(resources.get(baseline, {}) or {})
    group_resources = dict(group_cfg.get("resources", {}) or {})
    group_baseline_resources = dict(group_resources.get(baseline, {}) or {})
    if baseline == "vivid":
        return bool(
            group_cfg.get(
                "uses_official_inverse_observation_operator",
                group_baseline_resources.get(
                    "uses_official_inverse_observation_operator",
                    baseline_resources.get("uses_official_inverse_observation_operator", False),
                ),
            )
        )
    return bool(
        group_cfg.get(
            "uses_official_inverse_observation_operator",
            group_baseline_resources.get(
                "uses_official_inverse_observation_operator",
                baseline_resources.get("uses_official_inverse_observation_operator", False),
            ),
        )
    )


def _resolve_pdes(cfg: dict[str, Any], group_cfg: dict[str, Any], experiment_kind: str, ablation_factor: str) -> list[str]:
    if experiment_kind == "ablation" and _env_flag("FULL_ABLATION_ALL"):
        if ablation_factor == "time_varying_sensor_count":
            configured = list(ALL_PDES)
        else:
            configured = list(ALL_PDES)
    else:
        configured = _resolve_list(
            group_cfg.get("pdes", cfg.get("pdes", ALL_PDES)), ALL_PDES
        )
    if ablation_factor == "sparse_solution_multicondition":
        return _restrict_multicondition_pdes(configured)
    return configured


def _resolve_baselines(
    cfg: dict[str, Any],
    task_group: str,
    task: str,
    group_cfg: dict[str, Any],
    experiment_kind: str,
    ablation_factor: str,
) -> list[str]:
    configured: list[str]
    if experiment_kind == "ablation" and _env_flag("FULL_ABLATION_ALL"):
        if ablation_factor == "runtime_budget":
            budget_cfg = group_cfg.get("budget_by_baseline", {}) or group_cfg.get("runtime_budget_by_baseline", {}) or {}
            if budget_cfg:
                configured = list(budget_cfg)
            else:
                configured = _all_baselines_for_task(task)
        elif ablation_factor == "time_varying_sensor_count":
            configured = sorted(TIME_VARYING_SENSOR_BASELINES)
        else:
            configured = _all_baselines_for_task(task)
    else:
        configured = list(
            group_cfg.get("baselines", cfg.get("baselines", DEFAULT_BASELINES_BY_GROUP[task_group]))
        )

    requested_text = str(os.environ.get("BASELINES", "") or "").strip()
    if not requested_text:
        return configured
    requested = {item.strip().lower() for item in requested_text.split(",") if item.strip()}
    unknown = requested.difference(ALL_BASELINES)
    if unknown:
        raise ValueError(f"BASELINES contains unknown baselines: {sorted(unknown)}")
    return [baseline for baseline in configured if baseline.lower() in requested]


def _all_baselines_for_task(task: str) -> list[str]:
    if task in {"forward", "inverse"}:
        return ["fno", "deeponet", "ifno"]
    if task == "sparse_inverse":
        return ["recfno", "senseiver", "voronoicnn", "fno", "deeponet", "pinn_sparse", "pc_bnn", "pde_opt", "var4d", "vivid"]
    return list(ALL_BASELINES)


def _group_expansion(cfg: dict[str, Any], group_cfg: dict[str, Any], defaults: dict[str, Any], task_group: str, task: str) -> dict[str, list[Any]]:
    sparse = task.startswith("sparse")
    if sparse:
        sensor_counts = _env_list("SENSOR_COUNTS", _first_present(group_cfg, ["sensor_counts", "num_sensors", "sensor_count"], defaults["sensor_counts"]), int)
        sensor_modes = _env_list("SENSOR_MODES", _first_present(group_cfg, ["sensor_modes", "sensor_mode"], defaults["sensor_modes"]), str)
        noise_levels = _env_list("NOISE_LEVELS", _first_present(group_cfg, ["noise_levels", "noise_level"], defaults["noise_levels"]), float)
    else:
        sensor_counts = [0]
        sensor_modes = ["none"]
        noise_levels = [0.0]
    train_sizes = _env_list("TRAIN_SIZES", group_cfg.get("train_sizes", cfg.get("train_sizes", [group_cfg.get("train_size", defaults["train_size"])])), int)
    return {
        "sensor_counts": sensor_counts,
        "sensor_modes": sensor_modes,
        "noise_levels": noise_levels,
        "train_sizes": train_sizes,
    }


def _validate_group_design(
    cfg: dict[str, Any],
    group_cfg: dict[str, Any],
    defaults: dict[str, Any],
    experiment_kind: str,
    ablation_factor: str,
    task_group: str,
    task: str,
    allow_multi: bool,
) -> None:
    expansion = _group_expansion(cfg, group_cfg, defaults, task_group, task)
    if task == "sparse_solution_multicondition":
        if experiment_kind != "ablation" or ablation_factor != "sparse_solution_multicondition":
            raise ValueError(
                "sparse_solution_multicondition must use its independent ablation experiment kind/factor"
            )
        return
    lengths = {
        "num_sensors": len(set(expansion["sensor_counts"])),
        "sensor_mode": len(set(expansion["sensor_modes"])),
        "noise_level": len(set(expansion["noise_levels"])),
        "train_size": len(set(expansion["train_sizes"])),
    }
    if experiment_kind == "main":
        offenders = {field: length for field, length in lengths.items() if field in {"num_sensors", "sensor_mode", "noise_level"} and length > 1}
        if offenders:
            raise ValueError(f"main task_group {task_group!r} must use one sensor/noise/mode setting, got {offenders}")
        return

    budget_varied = _runtime_budget_varied_fields(group_cfg)
    multi_fields = {field for field, length in lengths.items() if length > 1}
    multi_fields.update(budget_varied)
    allowed = FACTOR_TO_VARIED_FIELDS[ablation_factor]
    illegal = multi_fields - allowed
    if illegal and not allow_multi:
        raise ValueError(
            f"ablation task_group {task_group!r} with factor {ablation_factor!r} varies unsupported fields "
            f"{sorted(illegal)}; set allow_multi_factor_grid: true only for explicit exhaustive debugging"
        )
    if not allow_multi and len(multi_fields & {"num_sensors", "sensor_mode", "noise_level", "train_size", "steps", "refine_steps", "particles"}) > len(allowed):
        active = sorted(multi_fields & {"num_sensors", "sensor_mode", "noise_level", "train_size", "steps", "refine_steps", "particles"})
        raise ValueError(f"ablation task_group {task_group!r} varies multiple factors: {active}")
    if ablation_factor == "runtime_budget":
        _validate_runtime_budget_group(group_cfg, task_group, allow_multi)
    elif not allow_multi and not (multi_fields & allowed):
        raise ValueError(f"ablation task_group {task_group!r} does not vary its declared factor {ablation_factor!r}")


def _runtime_budget_varied_fields(group_cfg: dict[str, Any]) -> set[str]:
    varied: set[str] = set()
    budget_cfg = group_cfg.get("budget_by_baseline", {}) or group_cfg.get("runtime_budget_by_baseline", {}) or {}
    for baseline_cfg in budget_cfg.values():
        for field in ("steps", "refine_steps", "particles"):
            if len(_as_list(baseline_cfg.get(field, []))) > 1:
                varied.add(field)
    for field in ("steps", "refine_steps", "particles"):
        if len(_as_list(group_cfg.get(field, []))) > 1:
            varied.add(field)
    return varied


def _validate_runtime_budget_group(group_cfg: dict[str, Any], task_group: str, allow_multi: bool) -> None:
    budget_cfg = group_cfg.get("budget_by_baseline", {}) or group_cfg.get("runtime_budget_by_baseline", {}) or {}
    if not budget_cfg and not any(group_cfg.get(field) is not None for field in ("steps", "refine_steps", "particles")):
        raise ValueError(f"runtime_budget ablation {task_group!r} must define budget_by_baseline or a budget field")
    if allow_multi:
        return
    for baseline, baseline_cfg in budget_cfg.items():
        varied = [field for field in ("steps", "refine_steps", "particles") if len(_as_list(baseline_cfg.get(field, []))) > 1]
        if len(varied) != 1:
            raise ValueError(f"runtime_budget baseline {baseline!r} must vary exactly one budget field, got {varied}")
    top_varied = [field for field in ("steps", "refine_steps", "particles") if len(_as_list(group_cfg.get(field, []))) > 1]
    if top_varied and len(top_varied) != 1:
        raise ValueError(f"runtime_budget task_group {task_group!r} must not cross budget fields, got {top_varied}")


def _budget_variants(
    cfg: dict[str, Any],
    group_cfg: dict[str, Any],
    baseline: str,
    ablation_factor: str,
    experiment_kind: str,
) -> list[dict[str, int]]:
    resources = _resource_config(cfg, baseline)
    base = {
        "steps": _method_steps(baseline, resources),
        "refine_steps": _method_refine_steps(baseline, resources),
        "particles": _method_particles(baseline, resources),
    }
    if experiment_kind != "ablation" or ablation_factor != "runtime_budget":
        return [base]

    budget_cfg = group_cfg.get("budget_by_baseline", {}) or group_cfg.get("runtime_budget_by_baseline", {}) or {}
    selected = budget_cfg.get(baseline)
    if selected is None:
        selected = {field: group_cfg[field] for field in ("steps", "refine_steps", "particles") if field in group_cfg}
    if not selected:
        return [base]
    varied = [field for field in ("steps", "refine_steps", "particles") if len(_as_list(selected.get(field, []))) > 1]
    if len(varied) > 1 and not _as_bool(cfg.get("allow_multi_factor_grid", False)):
        raise ValueError(f"runtime_budget baseline {baseline!r} crosses budget fields: {varied}")
    if not varied:
        merged = dict(base)
        for field in ("steps", "refine_steps", "particles"):
            if field in selected:
                merged[field] = int(selected[field])
        return [merged]
    field = varied[0]
    variants = []
    for value in _as_list(selected[field]):
        merged = dict(base)
        for fixed_field in ("steps", "refine_steps", "particles"):
            if fixed_field in selected and fixed_field != field and not isinstance(selected[fixed_field], list):
                merged[fixed_field] = int(selected[fixed_field])
        merged[field] = int(value)
        variants.append(merged)
    return variants


def _make_run_row(
    cfg: dict[str, Any],
    group_cfg: dict[str, Any],
    defaults: dict[str, Any],
    output_root: Path,
    matrix_name: str,
    experiment_kind: str,
    ablation_factor: str,
    comparison_track: str,
    capability,
    task_group: str,
    task: str,
    pde: str,
    baseline: str,
    seed: int,
    train_size: int,
    num_sensors: int,
    sensor_mode: str,
    noise_level: float,
    budget: dict[str, int],
    commit_hash: str,
    experiment_config_sha256: str,
    data_manifest_path: str,
    data_manifest_sha256: str,
    data_content_digest: str,
) -> dict[str, Any]:
    resources = _resource_config(cfg, baseline)
    val_size = _env_int("VAL_SIZE", int(group_cfg.get("val_size", defaults["val_size"])))
    test_size = _env_int("TEST_SIZE", int(group_cfg.get("test_size", defaults["test_size"])))
    train_shards = _env_int("TRAIN_SHARDS", int(group_cfg.get("train_shards", defaults["train_shards"])))
    batch_size = _env_int("BATCH_SIZE", int(group_cfg.get("batch_size", resources.get("batch_size", 16))))
    epochs = _env_int("EPOCHS", int(group_cfg.get("epochs", resources.get("epochs", 1 if baseline in PER_INSTANCE_BASELINES else 200))))
    load_full_trajectory = bool(
        group_cfg.get(
            "load_full_trajectory",
            resources.get("load_full_trajectory", defaults.get("load_full_trajectory", False)),
        )
    )
    if sensor_mode == "time_varying" or task_group == "time_varying_sensor_ablation":
        load_full_trajectory = True
    scalar_param_mode = _scalar_param_mode(defaults, pde, task)
    config_path = str(group_cfg.get("config", defaults["config"]))
    data_files = files_for_pde(defaults.get("data_files", {}), pde)
    if defaults.get("data_files") and data_files is None:
        raise ValueError(f"data_files_config does not define required PDE {pde!r}")
    method_config = load_config(config_path)
    physics_metric_mode = str(method_config.get("physics_metric_mode", "per_sample"))
    if physics_metric_mode not in {"per_sample", "per_batch"}:
        raise ValueError(
            "physics_metric_mode must be per_sample or per_batch, "
            f"got {physics_metric_mode!r} in {config_path}"
        )
    effective_method_config = resolve_method_config(
        method_config,
        baseline=baseline,
        pde=pde,
        epochs=epochs,
        steps=int(budget.get("steps", 0) or 0) or None,
        refine_steps=int(budget.get("refine_steps", 0) or 0) or None,
        particles=int(budget.get("particles", 0) or 0) or None,
        device=str(defaults["device"]),
        seed=seed,
    )
    reuse_ifno_forward = (
        baseline == "ifno"
        and task_group == "full_inverse_main"
        and task == "inverse"
    )
    execution_mode = str(
        group_cfg.get("execution_mode", "eval_only" if reuse_ifno_forward else "train")
    )
    if execution_mode not in {"train", "eval_only"}:
        raise ValueError(
            f"task group {task_group!r} has unsupported execution_mode={execution_mode!r}"
        )
    source_task = str(
        group_cfg.get("source_task", "forward" if reuse_ifno_forward else "")
    )
    row: dict[str, Any] = {
        "matrix_schema_version": MATRIX_SCHEMA_VERSION,
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "run_id": "",
        "run_fingerprint": "",
        "run_name": "",
        "execution_mode": execution_mode,
        "source_train_run_id": "pending" if execution_mode == "eval_only" else "",
        "source_train_run_fingerprint": DEPENDENCY_PENDING_SHA256 if execution_mode == "eval_only" else "",
        "source_train_task": source_task if execution_mode == "eval_only" else "",
        "source_train_baseline_code_sha256": (
            baseline_code_sha256(baseline, root=ROOT)
            if execution_mode == "eval_only"
            else ""
        ),
        "comparison_track": comparison_track,
        "experiment_kind": experiment_kind,
        "ablation_factor": ablation_factor,
        "task_group": task_group,
        "task": task,
        "task_protocol_version": str(
            group_cfg.get(
                "task_protocol_version",
                cfg.get("task_protocol_version", DEFAULT_TASK_PROTOCOL_VERSION),
            )
        ),
        "sensor_protocol_version": str(
            group_cfg.get(
                "sensor_protocol_version",
                cfg.get("sensor_protocol_version", DEFAULT_SENSOR_PROTOCOL_VERSION),
            )
        ),
        "data_manifest_sha256": data_manifest_sha256,
        "data_manifest_path": data_manifest_path,
        "capability_status": capability.support_status,
        "implementation_required": capability.implementation_required,
        "paper_table_eligible": bool(capability.paper_table_eligible),
        "unified_comparison_eligible": bool(capability.unified_comparison_eligible),
        "official_native_eligible": bool(capability.official_native_eligible),
        "pde": pde,
        "baseline": baseline,
        "seed": seed,
        "sensor_seed": _env_int("SENSOR_SEED", int(group_cfg.get("sensor_seed", cfg.get("sensor_seed", seed)))),
        "source_train_seed": seed if execution_mode == "eval_only" else "",
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "train_shards": train_shards,
        "num_sensors": int(num_sensors),
        "sensor_mode": sensor_mode,
        "sensor_budget_mode": str(
            group_cfg.get("sensor_budget_mode", cfg.get("sensor_budget_mode", "per_time"))
            if task.startswith("sparse")
            else "none"
        ),
        "noise_level": float(noise_level),
        "scalar_param_mode": scalar_param_mode,
        "physics_metric_mode": physics_metric_mode,
        "data_loading_mode": defaults["data_loading_mode"],
        "num_workers": int(defaults["num_workers"]),
        "pin_memory": bool(defaults["pin_memory"]),
        "persistent_workers": bool(defaults["persistent_workers"]),
        "prefetch_factor": int(defaults["prefetch_factor"]),
        "load_full_trajectory": load_full_trajectory,
        "batch_size": batch_size,
        "epochs": epochs,
        "steps": int(budget.get("steps", 0) or 0),
        "refine_steps": int(budget.get("refine_steps", 0) or 0),
        "particles": int(budget.get("particles", 0) or 0),
        "device": defaults["device"],
        "commit_hash": commit_hash,
        "config": config_path,
        "config_content_sha256": sha256_file(config_path, root=ROOT),
        "baseline_config_sha256": baseline_config_sha256(effective_method_config),
        "baseline_code_sha256": baseline_code_sha256(baseline, root=ROOT),
        "experiment_config_sha256": experiment_config_sha256,
        "data_content_sha256": data_content_digest,
        "checkpoint_path": "",
        "checkpoint_sha256": DEPENDENCY_PENDING_SHA256 if execution_mode == "eval_only" else "",
        "dependency_run_id": "",
        "dependency_pending": execution_mode == "eval_only",
        "dependency_reason": "source training checkpoint is not available" if execution_mode == "eval_only" else "",
        "output_dir": "",
        "log_dir": "",
        "status_file": "",
        "skip_reason": "",
    }
    if task == "sparse_solution_multicondition":
        probabilities = _validate_condition_probabilities_config(
            group_cfg.get(
                "condition_probabilities", cfg.get("condition_probabilities")
            )
        )
        condition_mode = str(group_cfg.get("condition_mode", cfg.get("condition_mode", "mixed")))
        if execution_mode == "train" and condition_mode != "mixed":
            raise ValueError(
                "sparse_solution_multicondition training rows must use condition_mode=mixed"
            )
        row.update(
            {
                "condition_mode": condition_mode,
                "condition_probabilities": probabilities,
                "evaluation_condition_modes": list(
                    cfg.get("evaluation_condition_modes", ["a_only", "u_only", "both"])
                ),
                "train_only": execution_mode == "train",
            }
        )
    if data_files is not None:
        row["data_files"] = data_files
    row["run_fingerprint"] = run_fingerprint(row)
    row["run_id"] = _run_id(row)
    row["run_name"] = _run_name(row)
    row["output_dir"] = str(_run_output_dir(output_root, matrix_name, row))
    row["log_dir"] = str(_run_log_dir(output_root, matrix_name, row))
    row["status_file"] = str(Path(row["output_dir"]) / "run.status.json")
    return row


def _resource_config(cfg: dict[str, Any], baseline: str) -> dict[str, Any]:
    resources = dict(cfg.get("resources", {}) or {})
    base_name = "per_instance_default" if baseline in PER_INSTANCE_BASELINES else "amortized_default"
    merged = dict(resources.get(base_name, {}) or {})
    merged.update(resources.get(baseline, {}) or {})
    return merged


def _method_steps(baseline: str, resources: dict[str, Any]) -> int:
    if baseline == "pinn_sparse":
        return _env_int("PINN_STEPS", int(resources.get("steps", 1000)))
    if baseline == "pde_opt":
        return _env_int("PDEOPT_STEPS", int(resources.get("steps", 500)))
    if baseline == "var4d":
        return _env_int("VAR4D_STEPS", int(resources.get("steps", 500)))
    if baseline == "pc_bnn":
        return _env_int("PCBNN_STEPS", int(resources.get("steps", 500)))
    return int(resources.get("steps", 0) or 0)


def _method_refine_steps(baseline: str, resources: dict[str, Any]) -> int:
    if baseline == "vivid":
        return _env_int("VIVID_REFINE_STEPS", int(resources.get("refine_steps", 1000)))
    return int(resources.get("refine_steps", 0) or 0)


def _method_particles(baseline: str, resources: dict[str, Any]) -> int:
    if baseline == "pc_bnn":
        return _env_int("PCBNN_PARTICLES", int(resources.get("particles", 8)))
    return int(resources.get("particles", 0) or 0)


def _scalar_param_mode(defaults: dict[str, Any], pde: str, task: str) -> str:
    explicit = str(defaults.get("scalar_param_mode", "") or "")
    if explicit:
        return explicit
    if pde in FUTURE_PDES and task in {"forward", "inverse", "sparse_inverse"}:
        return "materialize"
    return "metadata"


def _run_id(row: dict[str, Any]) -> str:
    fingerprint = str(row.get("run_fingerprint") or run_fingerprint(row))
    digest = fingerprint[:12]
    prefix = f"{row['task_group']}_{row['baseline']}_{row['pde']}_s{row['seed']}"
    return _safe_name(f"{prefix}_{digest}")


def _run_name(row: dict[str, Any]) -> str:
    base = f"{row['task_group']}/{row['baseline']}/{row['pde']}/seed={row['seed']}"
    if row["task"].startswith("sparse"):
        base += f"/sensors={row['num_sensors']}/{row['sensor_mode']}/noise={row['noise_level']}"
    if row["task"] == "sparse_solution_multicondition":
        base += f"/condition={row.get('condition_mode', 'mixed')}"
    if row["ablation_factor"] == "train_size":
        base += f"/train_size={row['train_size']}"
    if row["ablation_factor"] == "runtime_budget":
        budget = _budget_label(row)
        if budget:
            base += f"/{budget}"
    return base


def _run_output_dir(output_root: Path, matrix_name: str, row: dict[str, Any]) -> Path:
    if row["experiment_kind"] == "main":
        parts = [
            "runs",
            matrix_name,
            f"task_group={_safe_name(row['task_group'])}",
            f"pde={_safe_name(row['pde'])}",
            f"baseline={_safe_name(row['baseline'])}",
            f"seed={row['seed']}",
            f"run={row['run_id']}",
        ]
        return output_root.joinpath(*parts)
    ablation_name = _ablation_dir_name(row["ablation_factor"], matrix_name)
    parts = [
        "runs",
        "ablations",
        f"ablation={_safe_name(ablation_name)}",
        f"pde={_safe_name(row['pde'])}",
        f"baseline={_safe_name(row['baseline'])}",
        f"seed={row['seed']}",
        f"sensors={row['num_sensors']}",
        f"mode={_safe_name(row['sensor_mode'])}",
        f"noise={_safe_float(row['noise_level'])}",
    ]
    if row["task"] == "sparse_solution_multicondition":
        parts.append(f"condition={_safe_name(row.get('condition_mode', 'mixed'))}")
    if row["ablation_factor"] == "train_size":
        parts.append(f"train_size={row['train_size']}")
    if row["ablation_factor"] == "runtime_budget":
        budget = _budget_label(row)
        if budget:
            parts.append(_safe_name(budget))
    parts.append(f"run={row['run_id']}")
    return output_root.joinpath(*parts)


def _run_log_dir(output_root: Path, matrix_name: str, row: dict[str, Any]) -> Path:
    if row["experiment_kind"] == "main":
        return output_root / "logs" / matrix_name / f"task_group={_safe_name(row['task_group'])}" / f"run={row['run_id']}"
    ablation_name = _ablation_dir_name(row["ablation_factor"], matrix_name)
    return output_root / "logs" / "ablations" / f"ablation={_safe_name(ablation_name)}" / f"run={row['run_id']}"


def _ablation_dir_name(ablation_factor: str, matrix_name: str) -> str:
    return {
        "sensor_count": "sensor_count",
        "noise_level": "noise",
        "sensor_mode": "sensor_mode",
        "time_varying_sensor_count": "time_varying",
        "runtime_budget": "runtime_budget",
        "train_size": "train_size",
        "sparse_solution_multicondition": "sparse_solution_multicondition",
    }.get(ablation_factor, matrix_name)


def _budget_label(row: dict[str, Any]) -> str:
    baseline = str(row["baseline"])
    if baseline == "vivid" and int(row.get("refine_steps", 0) or 0) > 0:
        return f"refine_steps={int(row['refine_steps'])}"
    if baseline == "pc_bnn" and int(row.get("particles", 0) or 0) > 0 and int(row.get("steps", 0) or 0) <= 0:
        return f"particles={int(row['particles'])}"
    if int(row.get("steps", 0) or 0) > 0:
        return f"steps={int(row['steps'])}"
    if int(row.get("particles", 0) or 0) > 0:
        return f"particles={int(row['particles'])}"
    if int(row.get("refine_steps", 0) or 0) > 0:
        return f"refine_steps={int(row['refine_steps'])}"
    return ""


def _skipped_matrix_row(
    skip: dict[str, Any],
    defaults: dict[str, Any],
    output_root: Path,
    commit_hash: str,
    experiment_config_sha256: str,
    data_manifest_path: str,
    data_manifest_sha256: str,
    data_content_digest: str,
) -> dict[str, Any]:
    row = {field: "" for field in MATRIX_FIELDS}
    row.update(
        {
            "matrix_schema_version": MATRIX_SCHEMA_VERSION,
            "summary_schema_version": SUMMARY_SCHEMA_VERSION,
            "run_id": _safe_name(f"skipped_{skip['task_group']}_{skip['baseline']}_{skip['pde']}_{skip['sensor_mode']}"),
            "run_fingerprint": "",
            "execution_mode": "skipped",
            "source_train_run_id": "",
            "source_train_run_fingerprint": "",
            "comparison_track": skip.get("comparison_track", "unified_adapted"),
            "experiment_kind": skip.get("experiment_kind", ""),
            "ablation_factor": skip.get("ablation_factor", ""),
            "task_group": skip["task_group"],
            "task": skip["task"],
            "task_protocol_version": DEFAULT_TASK_PROTOCOL_VERSION,
            "sensor_protocol_version": DEFAULT_SENSOR_PROTOCOL_VERSION,
            "data_manifest_sha256": data_manifest_sha256,
            "data_manifest_path": data_manifest_path,
            "capability_status": skip.get("capability_status", "unsupported"),
            "implementation_required": skip.get("implementation_required", "unsupported"),
            "paper_table_eligible": bool(skip.get("paper_table_eligible", False)),
            "unified_comparison_eligible": bool(skip.get("unified_comparison_eligible", False)),
            "official_native_eligible": bool(skip.get("official_native_eligible", False)),
            "pde": skip["pde"],
            "baseline": skip["baseline"],
            "seed": 0,
            "sensor_seed": 0,
            "source_train_seed": "",
            "train_size": defaults["train_size"],
            "val_size": defaults["val_size"],
            "test_size": defaults["test_size"],
            "train_shards": defaults["train_shards"],
            "num_sensors": 0,
            "sensor_mode": skip["sensor_mode"],
            "noise_level": 0.0,
            "scalar_param_mode": "metadata",
            "physics_metric_mode": "per_sample",
            "data_loading_mode": defaults["data_loading_mode"],
            "num_workers": int(defaults["num_workers"]),
            "pin_memory": bool(defaults["pin_memory"]),
            "persistent_workers": bool(defaults["persistent_workers"]),
            "prefetch_factor": int(defaults["prefetch_factor"]),
            "load_full_trajectory": False,
            "batch_size": 0,
            "epochs": 0,
            "steps": 0,
            "refine_steps": 0,
            "particles": 0,
            "device": defaults["device"],
            "commit_hash": commit_hash,
            "config": defaults["config"],
            "config_content_sha256": "",
            "data_content_sha256": data_content_digest,
            "experiment_config_sha256": experiment_config_sha256,
            "checkpoint_path": "",
            "checkpoint_sha256": "",
            "output_dir": str(output_root / "skipped" / _safe_name(skip["task_group"])),
            "log_dir": str(output_root / "logs" / "skipped"),
            "status_file": "",
            "skip_reason": skip["reason"],
        }
    )
    return row


def _merge_skip(skipped: dict[tuple[Any, ...], dict[str, Any]], row: dict[str, Any]) -> None:
    key = (
        row["matrix_name"],
        row.get("experiment_kind", ""),
        row.get("ablation_factor", ""),
        row["task_group"],
        row["task"],
        row["pde"],
        row["baseline"],
        row["sensor_mode"],
        row["reason"],
    )
    if key not in skipped:
        skipped[key] = dict(row)
    else:
        skipped[key]["would_have_expanded"] = int(skipped[key]["would_have_expanded"]) + int(row["would_have_expanded"])


def _summary(
    rows: list[dict[str, Any]],
    skipped_rows: list[dict[str, Any]],
    matrix_name: str,
    experiment_kind: str,
    ablation_factor: str,
    comparison_track: str,
    experiment_config_sha256: str,
    data_manifest_path: str,
    data_manifest_sha256: str,
    data_content_digest: str,
) -> dict[str, Any]:
    active_rows = [row for row in rows if not row.get("skip_reason")]
    by_group = Counter(row["task_group"] for row in active_rows)
    by_pde = Counter(row["pde"] for row in active_rows)
    by_baseline = Counter(row["baseline"] for row in active_rows)
    by_group_baseline = Counter((row["task_group"], row["baseline"]) for row in active_rows)
    skipped_by_group = Counter()
    skipped_total_expanded = 0
    for row in skipped_rows:
        skipped_by_group[row["task_group"]] += int(row.get("would_have_expanded", 1))
        skipped_total_expanded += int(row.get("would_have_expanded", 1))
    varied_fields, fixed_fields = _field_variation(active_rows, ablation_factor)
    return {
        "matrix_schema_version": MATRIX_SCHEMA_VERSION,
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "matrix_name": matrix_name,
        "experiment_kind": experiment_kind,
        "ablation_factor": ablation_factor,
        "comparison_track": comparison_track,
        "experiment_config_sha256": experiment_config_sha256,
        "data_manifest_sha256": data_manifest_sha256,
        "data_content_sha256": data_content_digest,
        "data_manifest_path": data_manifest_path,
        "run_count": len(active_rows),
        "skipped_combo_count": len(skipped_rows),
        "skipped_expanded_count": skipped_total_expanded,
        "by_task_group": dict(sorted(by_group.items())),
        "by_pde": dict(sorted(by_pde.items())),
        "by_baseline": dict(sorted(by_baseline.items())),
        "varied_fields": varied_fields,
        "fixed_fields": fixed_fields,
        "skipped_by_task_group": dict(sorted(skipped_by_group.items())),
        "by_task_group_baseline": {
            f"{group}/{baseline}": count for (group, baseline), count in sorted(by_group_baseline.items())
        },
    }


def _field_variation(rows: list[dict[str, Any]], ablation_factor: str) -> tuple[dict[str, Any], dict[str, Any]]:
    varied: dict[str, Any] = {}
    fixed: dict[str, Any] = {}
    if not rows:
        return varied, fixed
    fields = list(SUMMARY_DESIGN_FIELDS)
    if ablation_factor != "runtime_budget":
        fields = [field for field in fields if field not in {"steps", "refine_steps", "particles"}]
    for field in fields:
        values = sorted({str(row.get(field, "")) for row in rows})
        if len(values) == 1:
            fixed[field] = values[0]
        else:
            varied[field] = {"count": len(values), "values": values[:30]}
    return varied, fixed


def _ensure_unique_run_ids(rows: list[dict[str, Any]]) -> None:
    ids = [row["run_id"] for row in rows if not row.get("skip_reason")]
    duplicates = [run_id for run_id, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"run_id collision detected: {duplicates[:10]}")


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MATRIX_FIELDS, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _resolve_list(value: Any, all_values: list[str]) -> list[str]:
    if isinstance(value, str) and value.lower() == "all":
        return list(all_values)
    if isinstance(value, str):
        return shlex.split(value)
    return list(value)


def _first_present(mapping: dict[str, Any], keys: list[str], default: Any) -> Any:
    for key in keys:
        if key in mapping:
            value = mapping[key]
            if key in {"num_sensors", "sensor_count", "sensor_mode", "noise_level"} and not isinstance(value, list):
                return [value]
            return value
    return default


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return shlex.split(value)
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value not in {None, ""} else int(default)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value in {None, ""}:
        return bool(default)
    return str(value).lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, default: Any, caster) -> list[Any]:
    raw = os.environ.get(name)
    if raw not in {None, ""}:
        return [caster(item) for item in shlex.split(raw)]
    if isinstance(default, str):
        return [caster(item) for item in shlex.split(default)]
    return [caster(item) for item in list(default)]


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0").lower() in {"1", "true", "yes", "on"}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "on"}


def _safe_name(value: Any) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "="} else "_" for ch in text)


def _safe_float(value: Any) -> str:
    return _safe_name(f"{float(value):g}")


def _short_list(values: list[Any], max_items: int = 12) -> str:
    text_values = [str(value) for value in values]
    if len(text_values) <= max_items:
        return ",".join(text_values)
    shown = ",".join(text_values[:max_items])
    return f"{shown},...({len(text_values)} total)"


def _output_paths(output_root: str | Path, matrix_name: str) -> dict[str, Path]:
    output_root = Path(output_root)
    matrix_dir = output_root / "matrices"
    return {
        "jsonl": matrix_dir / f"{matrix_name}.jsonl",
        "tsv": matrix_dir / f"{matrix_name}.tsv",
        "summary": matrix_dir / f"{matrix_name}_summary.json",
        "skipped": matrix_dir / f"{matrix_name}_skipped.jsonl",
        "skipped_global": output_root / "skipped_combinations.jsonl",
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.comparison_track:
        cfg["comparison_track"] = args.comparison_track
    matrix_name = args.matrix_name or str(cfg.get("name") or Path(args.config).stem)
    task_groups = list(cfg.get("task_groups", []))
    progress(
        f"[matrix start] config={args.config} matrix_name={matrix_name} output_root={args.output_root} "
        f"experiment_kind={cfg.get('experiment_kind', '')} "
        f"comparison_track={cfg.get('comparison_track', 'unified_adapted')} "
        f"task_groups={_short_list(task_groups)}"
    )
    existing_matrix_path = _output_paths(args.output_root, matrix_name)["jsonl"]
    resume_rows = _read_jsonl(existing_matrix_path)
    rows, skipped, summary = build_matrix(
        cfg,
        args.output_root,
        matrix_name,
        include_skipped=args.include_skipped,
        emit_progress=True,
        data_manifest=args.data_manifest or None,
        experiment_config_path=args.config,
        resume_rows=resume_rows,
    )
    write_outputs(rows, skipped, summary, args.output_root, matrix_name)
    output_paths = _output_paths(args.output_root, matrix_name)
    progress(
        f"[matrix complete] rows={len(rows)} active_rows={summary['run_count']} skipped_combo_count={len(skipped)} "
        f"skipped_expanded_count={summary.get('skipped_expanded_count', 0)} "
        f"by_task_group={json.dumps(summary.get('by_task_group', {}), sort_keys=True)}"
    )
    progress(
        f"[matrix outputs] jsonl={output_paths['jsonl']} tsv={output_paths['tsv']} summary={output_paths['summary']} "
        f"skipped={output_paths['skipped']} skipped_global={output_paths['skipped_global']}"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
