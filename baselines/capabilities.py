from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


ALL_PDES = [
    "darcy",
    "poisson",
    "helmholtz",
    "nsnonbounded",
    "burger",
    "reaction_diffusion",
    "shallow_water",
    "heat",
    "wave",
    "advection_diffusion",
    "steady_heat_conduction",
]

FUTURE_PDES = {"heat", "wave", "advection_diffusion", "steady_heat_conduction"}

STATIC_SPARSE_INVERSE_PDES = {"poisson", "helmholtz", "darcy", "steady_heat_conduction"}

TIME_DEPENDENT_PDES = {
    "nsnonbounded",
    "burger",
    "reaction_diffusion",
    "shallow_water",
    "heat",
    "wave",
    "advection_diffusion",
}

TIME_VARYING_DA_PDES = {"nsnonbounded", "burger", "reaction_diffusion", "shallow_water"}
VAR4D_VIVID_PDES = {"burger"}

ALL_BASELINES = [
    "deeponet",
    "fno",
    "ifno",
    "pc_bnn",
    "pde_opt",
    "pinn_sparse",
    "recfno",
    "senseiver",
    "var4d",
    "vivid",
    "voronoicnn",
]

SUPPORT_STATUSES = {"native", "official_adapter", "adapted", "unsupported"}
IMPLEMENTATION_REQUIREMENTS = {"official", "canonical_math", "adapted_allowed", "unsupported"}

SOURCE_KEYS = {
    "fno": "fno",
    "deeponet": "deeponet",
    "ifno": "ifno",
    "recfno": "recfno",
    "senseiver": "senseiver",
    "voronoicnn": "voronoicnn",
    "pinn_sparse": "pinn_sparse",
    "pc_bnn": "pc_bnn",
    "pde_opt": "pde_opt",
    "var4d": "var4d",
    "vivid": "vivid",
}

CITATION_KEYS = {
    "fno": "li2021fno",
    "deeponet": "lu2021deeponet",
    "ifno": "long2025ifno",
    "recfno": "zhao2023recfno",
    "senseiver": "santos2023senseiver",
    "voronoicnn": "fukami2021voronoicnn",
    "pinn_sparse": "raissi2019pinn_deepxde",
    "pc_bnn": "sunwang_pcbnn",
    "pde_opt": "pde_constrained_optimization",
    "var4d": "4dvar_canonical",
    "vivid": "vivid_invobs",
}


@dataclass(frozen=True)
class Capability:
    baseline: str
    pde: str
    task: str
    sensor_mode: str
    support_status: str
    implementation_required: str
    task_family: str
    reason: str
    citation_key: str
    source_key: str
    notes_for_paper: str
    official_architecture_allowed: bool
    official_aligned_allowed: bool
    eligible_implementation_modes: tuple[str, ...]
    paper_table_eligible: bool
    unified_comparison_eligible: bool
    official_native_eligible: bool

    @property
    def unsupported_reason(self) -> str:
        return self.reason if self.support_status == "unsupported" else ""

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["unsupported_reason"] = self.unsupported_reason
        return row


def task_family_for(task: str, sensor_mode: str = "", task_group: str = "") -> str:
    task = str(task).lower()
    sensor_mode = str(sensor_mode or "").lower()
    task_group = str(task_group or "").lower()
    if task_group in {"time_varying", "time_varying_da"} or task_group.startswith("time_varying") or sensor_mode == "time_varying":
        return "time_varying_da"
    if task == "forward":
        return "full_forward"
    if task == "inverse":
        return "full_inverse"
    if task == "sparse_solution_multicondition":
        return "sparse_solution_multicondition"
    if task in {"sparse_solution", "sparse_reconstruction"}:
        return "sparse_reconstruction"
    if task == "sparse_inverse":
        return "sparse_inverse"
    if task == "sparse_forward":
        return "sparse_forward"
    return task


def resolve_capability(
    baseline: str,
    pde: str,
    task: str,
    sensor_mode: str = "",
    task_group: str = "",
    *,
    load_full_trajectory: bool | None = None,
    train_inverse_operator: bool | None = None,
    uses_official_inverse_observation_operator: bool | None = None,
) -> Capability:
    baseline = str(baseline).lower()
    pde = str(pde).lower()
    task = str(task).lower()
    sensor_mode = str(sensor_mode or "").lower()
    task_group = str(task_group or "").lower()
    family = task_family_for(task, sensor_mode, task_group)
    if (
        baseline in {"var4d", "vivid"}
        and pde == "burger"
        and task in {"sparse_solution", "sparse_reconstruction"}
        and load_full_trajectory is not False
    ):
        # These methods always operate on the complete Burgers assimilation
        # window, even when a direct CLI invocation omits the matrix task-group
        # label and uses random_per_sample sensors.
        family = "time_varying_da"

    if pde in {"diffusionpde", "diffusion_pde", "diffusion"}:
        return _cap(
            baseline,
            pde,
            task,
            sensor_mode,
            "unsupported",
            "unsupported",
            family,
            "DiffusionPDE is not part of the external-baseline comparison matrix",
        )
    if baseline in {"fm4pde", "fm4pde_debug", "fm4pde-ablation", "fm4pde_ablation"}:
        return _cap(
            baseline,
            pde,
            task,
            sensor_mode,
            "unsupported",
            "unsupported",
            family,
            "FM4PDE main-model/internal-ablation runs are outside this external-baseline matrix",
        )
    if pde not in set(ALL_PDES):
        return _cap(baseline, pde, task, sensor_mode, "unsupported", "unsupported", family, f"unknown PDE '{pde}'")
    if baseline not in set(ALL_BASELINES):
        return _cap(baseline, pde, task, sensor_mode, "unsupported", "unsupported", family, f"unknown baseline '{baseline}'")

    if task == "sparse_solution_multicondition":
        if pde == "burger":
            return _cap(
                baseline,
                pde,
                task,
                sensor_mode,
                "unsupported",
                "unsupported",
                family,
                "sparse_solution_multicondition does not support Burgers trajectory semantics",
            )
        if pde not in {"poisson", "helmholtz", "darcy", "nsnonbounded"}:
            return _cap(
                baseline,
                pde,
                task,
                sensor_mode,
                "unsupported",
                "unsupported",
                family,
                "sparse_solution_multicondition is scoped to Poisson, Helmholtz, Darcy, and NSnonbounded",
            )
        if baseline not in {"recfno", "senseiver", "voronoicnn"}:
            return _cap(
                baseline,
                pde,
                task,
                sensor_mode,
                "unsupported",
                "unsupported",
                family,
                "sparse_solution_multicondition is defined only for RecFNO, Senseiver, and VoronoiCNN",
            )
        return _cap(
            baseline,
            pde,
            task,
            sensor_mode,
            "adapted",
            "adapted_allowed",
            family,
            "FM4PDE supplies a modality-presence multicondition adapter around the baseline reconstruction backbone",
            "This is an independent multicondition task adapter and must not be described as an official-native task protocol.",
            eligible=False,
            unified_comparison_eligible=False,
            official_native_eligible=False,
        )

    if pde == "burger" and task == "sparse_inverse":
        return _cap(
            baseline,
            pde,
            task,
            sensor_mode,
            "unsupported",
            "unsupported",
            family,
            "Burger sparse_inverse is excluded: the sensor-only protocol keeps sparse trajectory reconstruction, while full inverse is u(T)->u(0)",
        )

    if family == "time_varying_da":
        return _time_varying_capability(
            baseline,
            pde,
            task,
            sensor_mode,
            family,
            load_full_trajectory=load_full_trajectory,
            train_inverse_operator=train_inverse_operator,
            uses_official_inverse_observation_operator=uses_official_inverse_observation_operator,
        )

    if baseline == "fno":
        if family == "full_forward":
            return _cap(
                baseline,
                pde,
                task,
                sensor_mode,
                "adapted",
                "adapted_allowed",
                family,
                "NeuralOperator 2.0 supplies the FNO component, while this repository supplies unified data, normalization, loss, and training",
                "Report as NeuralOperator FNO component + unified adapted training, not a classic-paper end-to-end reproduction.",
                eligible=False,
            )
        if family == "full_inverse":
            return _cap(
                baseline,
                pde,
                task,
                sensor_mode,
                "adapted",
                "adapted_allowed",
                family,
                "FNO inverse is a supervised inverse-operator adaptation, not a native FNO claim",
                "Report explicitly as an FNO-inverse adaptation.",
                eligible=False,
            )
        if family in {"sparse_reconstruction", "sparse_inverse"}:
            return _cap(
                baseline,
                pde,
                task,
                sensor_mode,
                "unsupported",
                "unsupported",
                family,
                "vanilla FNO has no native sparse-sensor reconstruction/inverse interface",
                "A masked-grid FNO can be run only as an explicitly named adaptation.",
            )

    if baseline == "deeponet":
        if family == "full_forward":
            return _cap(
                baseline,
                pde,
                task,
                sensor_mode,
                "adapted",
                "adapted_allowed",
                family,
                "DeepXDE supplies the CartesianProd network component, while this repository supplies unified training and data",
                "Report as DeepXDE component + unified adapted training, not an official example reproduction.",
                eligible=False,
            )
        if family == "full_inverse":
            return _cap(
                baseline,
                pde,
                task,
                sensor_mode,
                "adapted",
                "adapted_allowed",
                family,
                "DeepONet inverse is a supervised inverse-operator adaptation",
                "Report as an adaptation, not native DeepONet inverse ability.",
                eligible=False,
            )
        if family == "sparse_reconstruction":
            return _cap(
                baseline,
                pde,
                task,
                sensor_mode,
                "adapted",
                "adapted_allowed",
                family,
                "sensor values as branch input are a DeepONet-sensor adaptation",
                "Do not mix this adapted result into the native sparse reconstruction main table.",
                eligible=False,
            )
        if family == "sparse_inverse":
            return _cap(
                baseline,
                pde,
                task,
                sensor_mode,
                "unsupported",
                "unsupported",
                family,
                "DeepONet sparse inverse is not a standard native baseline in this matrix",
            )

    if baseline == "ifno":
        if family in {"full_forward", "full_inverse"}:
            return _cap(
                baseline,
                pde,
                task,
                sensor_mode,
                "adapted",
                "adapted_allowed",
                family,
                "The iFNO adapter preserves vendored FNOBlocks inside bidirectional coupling, the official VAE topology, posterior-mean inverse inference, and iFNO/VAE/joint training on FM4PDE fields",
                "Report as an official-training-aligned iFNO task adaptation; it is not a native official dataset run.",
                eligible=False,
            )
        return _cap(
            baseline,
            pde,
            task,
            sensor_mode,
            "unsupported",
            "unsupported",
            family,
            "iFNO sparse reconstruction/inverse is not native without an official posterior/sparse inference path",
        )

    if baseline == "recfno":
        if family in {"sparse_reconstruction", "sparse_forward"}:
            return _cap(
                baseline,
                pde,
                task,
                sensor_mode,
                "adapted",
                "adapted_allowed",
                family,
                "RecFNO contributes its VoronoiFNO2d component, but this repository uses a unified data/training adapter",
                "Label as an official component with unified adapted training; do not claim an end-to-end official reproduction.",
                eligible=False,
            )
        if family == "sparse_inverse":
            return _cap(
                baseline,
                pde,
                task,
                sensor_mode,
                "adapted",
                "adapted_allowed",
                family,
                "RecFNO sparse inverse changes the paper task and uses the component through a unified supervised adapter",
                "Report as an adapted data-interface task, not a native RecFNO result.",
                eligible=False,
            )
        return _cap(
            baseline,
            pde,
            task,
            sensor_mode,
            "unsupported",
            "unsupported",
            family,
            "RecFNO is not a full forward/inverse operator-learning baseline",
        )

    if baseline == "senseiver":
        if family in {"sparse_reconstruction", "sparse_forward"}:
            return _cap(
                baseline,
                pde,
                task,
                sensor_mode,
                "adapted",
                "adapted_allowed",
                family,
                "Senseiver contributes official Encoder/Decoder and Fourier coordinate components under unified training",
                "Label as an official component with unified adapted training.",
                eligible=False,
            )
        if family == "sparse_inverse":
            return _cap(
                baseline,
                pde,
                task,
                sensor_mode,
                "adapted",
                "adapted_allowed",
                family,
                "Senseiver sparse inverse changes the native reconstruction target through a unified supervised adapter",
                "Report as an adapted data-interface task, not a native Senseiver result.",
                eligible=False,
            )
        return _cap(
            baseline,
            pde,
            task,
            sensor_mode,
            "unsupported",
            "unsupported",
            family,
            "Senseiver is not a full supervised forward/inverse operator baseline here",
        )

    if baseline == "voronoicnn":
        if family in {"sparse_reconstruction", "sparse_forward"}:
            return _cap(
                baseline,
                pde,
                task,
                sensor_mode,
                "adapted",
                "adapted_allowed",
                family,
                "VoronoiCNN uses a PyTorch reimplementation of the published convolution stack under unified training",
                "Label as an official-architecture reimplementation with adapted training and disclose the configured width.",
                eligible=False,
            )
        if family == "sparse_inverse":
            return _cap(
                baseline,
                pde,
                task,
                sensor_mode,
                "adapted",
                "adapted_allowed",
                family,
                "VoronoiCNN sparse inverse changes the published reconstruction target and uses a PyTorch architecture reimplementation",
                "Report as an adapted data-interface task.",
                eligible=False,
            )
        return _cap(
            baseline,
            pde,
            task,
            sensor_mode,
            "unsupported",
            "unsupported",
            family,
            "VoronoiCNN is not a full forward/inverse operator-learning baseline",
        )

    if baseline == "pinn_sparse":
        if family == "sparse_reconstruction":
            return _cap(
                baseline,
                pde,
                task,
                sensor_mode,
                "unsupported",
                "unsupported",
                family,
                "sensor-only sparse_solution forbids the hidden full source/coefficient/initial fields required by this PINN objective",
                "Provide the same explicit PDE context to every method in a separately named protocol before enabling it.",
            )
        if family == "sparse_forward":
            if pde == "steady_heat_conduction":
                return _cap(
                    baseline,
                    pde,
                    task,
                    sensor_mode,
                    "adapted",
                    "canonical_math",
                    family,
                    "steady heat conduction uses the local generator-aligned discrete PINN objective, not the DeepXDE-native PDE adapter",
                    "Report separately from the Poisson/Helmholtz/Darcy DeepXDE training-API runs.",
                )
            if pde in STATIC_SPARSE_INVERSE_PDES:
                return _cap(
                    baseline,
                    pde,
                    task,
                    sensor_mode,
                    "official_adapter",
                    "canonical_math",
                    family,
                    "PINN is native for per-instance PDE fitting from sparse observations and uses DeepXDE PDE/PointSetBC/autodiff training",
                    "Report as a DeepXDE-native training API with an FM4PDE joint-field task adapter.",
                )
            return _cap(
                baseline,
                pde,
                task,
                sensor_mode,
                "unsupported",
                "unsupported",
                family,
                "time-dependent sparse forward is disabled until the source/coefficient residual is explicit",
            )
        if family == "sparse_inverse":
            if pde == "steady_heat_conduction":
                return _cap(
                    baseline,
                    pde,
                    task,
                    sensor_mode,
                    "adapted",
                    "canonical_math",
                    family,
                    "steady heat conduction uses the local generator-aligned discrete PINN objective, not the DeepXDE-native PDE adapter",
                    "Report separately from the Poisson/Helmholtz/Darcy DeepXDE training-API runs.",
                )
            if pde in STATIC_SPARSE_INVERSE_PDES:
                return _cap(
                    baseline,
                    pde,
                    task,
                    sensor_mode,
                    "official_adapter",
                    "canonical_math",
                    family,
                    "static PDE inverse uses a DeepXDE joint (solution, unknown) network, PointSetBC observations, and autodiff residual",
                    "Report as a DeepXDE-native training API with an FM4PDE joint-field task adapter.",
                )
            return _cap(
                baseline,
                pde,
                task,
                sensor_mode,
                "unsupported",
                "unsupported",
                family,
                "time-dependent sparse inverse is disabled until the unknown parameter/initial-state residual is explicit",
            )
        return _cap(
            baseline,
            pde,
            task,
            sensor_mode,
            "unsupported",
            "unsupported",
            family,
            "PINN-Sparse is a per-instance sparse observation baseline, not supervised full operator learning",
        )

    if baseline == "pc_bnn":
        if family == "sparse_reconstruction":
            if pde == "shallow_water":
                return _cap(
                    baseline,
                    pde,
                    task,
                    sensor_mode,
                    "adapted",
                    "adapted_allowed",
                    family,
                    "the official PC-BNN outputs incompressible-flow (u,v,p), not shallow-water state channels",
                    "Do not treat channel-count coincidence as official task alignment.",
                    eligible=False,
                    unified_comparison_eligible=False,
                )
            return _cap(
                baseline,
                pde,
                task,
                sensor_mode,
                "adapted",
                "adapted_allowed",
                family,
                "official PC-BNN channel/PDE assumptions are not matched by the default FM4PDE scalar-field tasks",
                "Generic SVGD particles are a separate debug method and are excluded from unified comparison matrices.",
                eligible=False,
                unified_comparison_eligible=False,
            )
        if family in {"sparse_forward", "sparse_inverse"}:
            if pde in STATIC_SPARSE_INVERSE_PDES:
                return _cap(
                    baseline,
                    pde,
                    task,
                    sensor_mode,
                    "adapted",
                    "adapted_allowed",
                    family,
                    "PC-BNN-adapted preserves the official Swish particles, hierarchical priors, Gamma initialization, SVGD-transformed per-particle Adam, and likelihood structure behind a static-PDE adapter",
                    "Report explicitly as PC-BNN official-training-aligned adaptation; it is eligible for the unified comparison but not an official-native dataset run.",
                    eligible=False,
                    unified_comparison_eligible=True,
                )
            return _cap(
                baseline,
                pde,
                task,
                sensor_mode,
                "unsupported",
                "unsupported",
                family,
                "PC-BNN sparse forward/inverse is enabled only for static PDEs with an explicit joint source/solution residual",
            )
        return _cap(
            baseline,
            pde,
            task,
            sensor_mode,
            "unsupported",
            "unsupported",
            family,
            "PC-BNN is not a supervised full-operator baseline",
        )

    if baseline == "pde_opt":
        if family == "sparse_reconstruction":
            return _cap(
                baseline,
                pde,
                task,
                sensor_mode,
                "unsupported",
                "unsupported",
                family,
                "sensor-only sparse_solution forbids the hidden full source/coefficient/initial fields required by this PDE optimization objective",
                "Provide the same explicit PDE context to every method in a separately named protocol before enabling it.",
            )
        if family == "sparse_forward":
            if pde in STATIC_SPARSE_INVERSE_PDES:
                return _cap(
                    baseline,
                    pde,
                    task,
                    sensor_mode,
                    "native",
                    "canonical_math",
                    family,
                    "PDE-constrained optimization is a canonical per-instance sparse reconstruction/forward baseline",
                    "No single official repository is claimed.",
                )
            return _cap(
                baseline,
                pde,
                task,
                sensor_mode,
                "unsupported",
                "unsupported",
                family,
                "time-dependent sparse forward is disabled until the source/coefficient residual is explicit",
            )
        if family == "sparse_inverse":
            if pde in STATIC_SPARSE_INVERSE_PDES:
                return _cap(
                    baseline,
                    pde,
                    task,
                    sensor_mode,
                    "native",
                    "canonical_math",
                    family,
                    "static PDE sparse inverse is supported by joint coefficient/source and solution optimization",
                    "Canonical math baseline; no official code claim.",
                )
            return _cap(
                baseline,
                pde,
                task,
                sensor_mode,
                "unsupported",
                "unsupported",
                family,
                "time-dependent sparse inverse is disabled until an explicit trajectory/parameter objective is implemented",
            )
        return _cap(
            baseline,
            pde,
            task,
            sensor_mode,
            "unsupported",
            "unsupported",
            family,
            "PDE-Opt is per-instance optimization, not supervised full operator learning",
        )

    if baseline in {"var4d", "vivid"}:
        if family == "sparse_reconstruction" and pde in TIME_DEPENDENT_PDES:
            label = "two_level_surrogate" if pde not in TIME_VARYING_DA_PDES else "state_or_endpoint_surrogate"
            return _cap(
                baseline,
                pde,
                task,
                sensor_mode,
                "adapted",
                "adapted_allowed",
                family,
                f"{baseline} without time-varying trajectory observations is a {label}, not native DA",
                "Endpoint-only adaptation; use sensor_mode=time_varying with --load-full-trajectory for trajectory DA.",
                eligible=False,
            )
        return _cap(
            baseline,
            pde,
            task,
            sensor_mode,
            "unsupported",
            "unsupported",
            family,
            f"{baseline} only enters the main comparison in time-varying data-assimilation settings",
        )

    return _cap(baseline, pde, task, sensor_mode, "unsupported", "unsupported", family, "unhandled baseline/task combination")


def paper_table_eligible(
    capability: Capability,
    implementation_mode_effective: str = "",
    backend_info: dict[str, Any] | None = None,
) -> bool:
    backend_info = backend_info or {}
    mode = str(backend_info.get("implementation_mode_effective", implementation_mode_effective) or "").lower()
    if not capability.paper_table_eligible:
        return False
    if capability.support_status not in {"native", "official_adapter"}:
        return False
    if bool(backend_info.get("fallback_used", False)):
        return False
    if mode not in set(capability.eligible_implementation_modes):
        return False
    adapter_status = str(backend_info.get("adapter_status", "") or "").lower()
    if any(token in adapter_status for token in ("style", "toy", "debug", "local", "fallback", "surrogate", "target_change", "adapted")):
        return False
    if capability.implementation_required == "official":
        if mode == "official":
            return bool(backend_info.get("official_import_success", False))
        if mode == "official_architecture":
            return bool(capability.official_architecture_allowed) and bool(backend_info.get("official_reimplementation_success", False))
        if mode == "official_aligned":
            return bool(capability.official_aligned_allowed) and bool(backend_info.get("official_reimplementation_success", False))
        return False
    if capability.implementation_required == "canonical_math":
        return mode == "canonical_math"
    return False


def iter_capability_matrix(
    pdes: Iterable[str] | None = None,
    baselines: Iterable[str] | None = None,
    tasks: Iterable[str] | None = None,
    sensor_modes: Iterable[str] | None = None,
) -> list[Capability]:
    pdes = list(pdes or ALL_PDES)
    baselines = list(baselines or ALL_BASELINES)
    tasks = list(tasks or ["forward", "inverse", "sparse_solution", "sparse_inverse"])
    sensor_modes = list(sensor_modes or ["none", "random_per_sample", "grid", "fixed", "time_varying", "time_slices_per_sample"])
    rows: list[Capability] = []
    for baseline in baselines:
        for pde in pdes:
            for task in tasks:
                modes = ["none"] if not str(task).startswith("sparse") else sensor_modes
                for sensor_mode in modes:
                    load_full_trajectory = sensor_mode == "time_varying"
                    rows.append(
                        resolve_capability(
                            baseline,
                            pde,
                            task,
                            "" if sensor_mode == "none" else sensor_mode,
                            "time_varying" if sensor_mode == "time_varying" else "",
                            load_full_trajectory=load_full_trajectory,
                            train_inverse_operator=True,
                            uses_official_inverse_observation_operator=True,
                        )
                    )
    return rows


def write_capability_matrix(output: str | Path, capabilities: Iterable[Capability] | None = None) -> tuple[Path, Path]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    capabilities = list(capabilities or iter_capability_matrix())
    rows = [cap.to_row() for cap in capabilities]
    csv_path = output / "baseline_capability_matrix.csv"
    json_path = output / "baseline_capability_matrix.json"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    return csv_path, json_path


def _time_varying_capability(
    baseline: str,
    pde: str,
    task: str,
    sensor_mode: str,
    family: str,
    *,
    load_full_trajectory: bool | None,
    train_inverse_operator: bool | None,
    uses_official_inverse_observation_operator: bool | None,
) -> Capability:
    if task not in {"sparse_solution", "sparse_reconstruction"}:
        return _cap(
            baseline,
            pde,
            task,
            sensor_mode,
            "unsupported",
            "unsupported",
            family,
            "time-varying DA is defined for sparse trajectory/state observation tasks only",
        )
    if baseline in {"var4d", "vivid"} and pde not in VAR4D_VIVID_PDES:
        return _cap(
            baseline,
            pde,
            task,
            sensor_mode,
            "unsupported",
            "unsupported",
            family,
            "the registered Var4D/VIVID implementations are scoped only to Burgers sparse trajectory reconstruction",
            "Legacy NS/reaction-diffusion/shallow-water experiment entries remain removed.",
        )
    if pde not in TIME_VARYING_DA_PDES:
        if baseline in {"var4d", "vivid"} and pde in TIME_DEPENDENT_PDES:
            return _cap(
                baseline,
                pde,
                task,
                sensor_mode,
                "adapted",
                "adapted_allowed",
                family,
                "endpoint-only two-level surrogate is not full 4D-Var/VIVID",
                "Report explicitly as a two_level_surrogate adaptation.",
                eligible=False,
            )
        return _cap(
            baseline,
            pde,
            task,
            sensor_mode,
            "unsupported",
            "unsupported",
            family,
            "time-varying DA main table requires explicit multi-time observations/trajectory data",
        )
    if load_full_trajectory is False:
        if baseline in {"var4d", "vivid"}:
            return _cap(
                baseline,
                pde,
                task,
                sensor_mode,
                "unsupported",
                "unsupported",
                family,
                "the Burgers Var4D/VIVID methods require the complete T x X trajectory contract",
                "Endpoint/two-level legacy entries were removed.",
            )
        return _cap(
            baseline,
            pde,
            task,
            sensor_mode,
            "adapted",
            "adapted_allowed",
            family,
            "endpoint-only run is a surrogate and cannot be reported as time-varying DA",
            "Set --load-full-trajectory and sensor_mode=time_varying for main-table eligibility.",
            eligible=False,
        )
    if baseline == "senseiver":
        return _cap(
            baseline,
            pde,
            task,
            sensor_mode,
            "adapted",
            "adapted_allowed",
            family,
            "Senseiver supports irregular time-varying coordinates through official components and a unified adapter",
            "Label as official components with adapted data/training, not an end-to-end native run.",
            eligible=False,
        )
    if baseline == "var4d":
        return _cap(
            baseline,
            pde,
            task,
            sensor_mode,
            "native",
            "canonical_math",
            family,
            "canonical strong-constraint Burgers 4D-Var optimizes only the initial state and propagates the complete window through differentiable dynamics",
            "The pseudo-spectral Burgers propagator and sparse-observation background are task adapters; no end-to-end official-native code claim is made.",
            eligible=True,
            official_native_eligible=False,
        )
    if baseline == "vivid":
        if train_inverse_operator is False:
            return _cap(
                baseline,
                pde,
                task,
                sensor_mode,
                "unsupported",
                "unsupported",
                family,
                "VIVID requires its supervised VCNN inverse operator to be trained or loaded from a checkpoint",
            )
        if uses_official_inverse_observation_operator is False:
            return _cap(
                baseline,
                pde,
                task,
                sensor_mode,
                "unsupported",
                "unsupported",
                family,
                "the official-core VIVID recipe requires the audited VIVID VCNN inverse operator",
            )
        return _cap(
            baseline,
            pde,
            task,
            sensor_mode,
            "official_adapter",
            "official",
            family,
            "the Burgers adapter ports the vendored VIVID VCNN architecture/training and three-term L-BFGS-B 3D-Var objective",
            "Report as an official-architecture VIVID task adapter: the T x X Burgers solution is the 2-D state and the dense covariance uses a matrix-free circulant embedding.",
            eligible=True,
            official_architecture_allowed=True,
            eligible_implementation_modes=("official_architecture",),
            official_native_eligible=False,
        )
    return _cap(
        baseline,
        pde,
        task,
        sensor_mode,
        "unsupported",
        "unsupported",
        family,
        f"{baseline} is not a native time-varying data-assimilation baseline",
    )


def _cap(
    baseline: str,
    pde: str,
    task: str,
    sensor_mode: str,
    support_status: str,
    implementation_required: str,
    task_family: str,
    reason: str,
    notes_for_paper: str = "",
    *,
    eligible: bool | None = None,
    official_architecture_allowed: bool = False,
    official_aligned_allowed: bool = False,
    eligible_implementation_modes: Iterable[str] | None = None,
    unified_comparison_eligible: bool = True,
    official_native_eligible: bool = False,
) -> Capability:
    if support_status not in SUPPORT_STATUSES:
        raise ValueError(f"Invalid support_status {support_status!r}")
    if implementation_required not in IMPLEMENTATION_REQUIREMENTS:
        raise ValueError(f"Invalid implementation_required {implementation_required!r}")
    if eligible is None:
        eligible = support_status in {"native", "official_adapter"} and implementation_required in {"official", "canonical_math"}
    if eligible_implementation_modes is None:
        if not eligible:
            modes: tuple[str, ...] = ()
        elif implementation_required == "official":
            modes_list = ["official"]
            if official_architecture_allowed:
                modes_list.append("official_architecture")
            if official_aligned_allowed:
                modes_list.append("official_aligned")
            modes = tuple(modes_list)
        elif implementation_required == "canonical_math":
            modes = ("canonical_math",)
        else:
            modes = ()
    else:
        modes = tuple(str(mode).lower() for mode in eligible_implementation_modes)
    return Capability(
        baseline=baseline,
        pde=pde,
        task=task,
        sensor_mode=sensor_mode or "none",
        support_status=support_status,
        implementation_required=implementation_required,
        task_family=task_family,
        reason=reason,
        citation_key=CITATION_KEYS.get(baseline, ""),
        source_key=SOURCE_KEYS.get(baseline, ""),
        notes_for_paper=notes_for_paper,
        official_architecture_allowed=bool(official_architecture_allowed),
        official_aligned_allowed=bool(official_aligned_allowed),
        eligible_implementation_modes=modes,
        paper_table_eligible=bool(eligible),
        unified_comparison_eligible=bool(unified_comparison_eligible),
        official_native_eligible=bool(official_native_eligible),
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Dump FM4PDE native baseline capability matrix")
    parser.add_argument("--output", default="outputs/baselines/capability_matrix")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    csv_path, json_path = write_capability_matrix(args.output)
    print(json.dumps({"csv": str(csv_path), "json": str(json_path)}, indent=2))


if __name__ == "__main__":
    main()
