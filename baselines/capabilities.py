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
    if task in {"sparse_solution", "sparse_reconstruction"}:
        return "sparse_reconstruction"
    if task == "sparse_inverse":
        return "sparse_inverse"
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
                "native",
                "official",
                family,
                "vanilla FNO is a full-grid supervised forward operator baseline",
                "Use neuraloperator or a recorded vendored zongyi-li/fourier_neural_operator FNO for paper mode; RecFNO components and local compact FNO are adapted supplement only.",
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
                "Report only as FNO-inverse adaptation in supplement.",
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
                "A masked-grid FNO can be run only as an explicitly named adapted supplement.",
            )

    if baseline == "deeponet":
        if family == "full_forward":
            return _cap(
                baseline,
                pde,
                task,
                sensor_mode,
                "native",
                "official",
                family,
                "DeepONet is a supervised operator-learning baseline for full forward maps",
                "Use DeepXDE or official DeepONet components in paper mode.",
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
                "Report as adapted supplement, not native DeepONet inverse ability.",
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
                "native",
                "official",
                family,
                "iFNO is defined for full forward and inverse operator learning",
                "Use direct iFNO components if importable; otherwise use the disclosed official-aligned invertible FNO architecture reimplementation.",
                official_architecture_allowed=True,
                official_aligned_allowed=True,
                eligible_implementation_modes=("official", "official_architecture", "official_aligned"),
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
        if family == "sparse_reconstruction":
            return _cap(
                baseline,
                pde,
                task,
                sensor_mode,
                "official_adapter",
                "official",
                family,
                "RecFNO is native for sparse-sensor global field reconstruction with mask/Voronoi embedding",
                "Use vendored RecFNO VoronoiFNO2d/model components in paper mode.",
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
                "changing the supervised target to coefficient/source is a RecFNO adaptation",
                "Report only in supplement as adapted sparse inverse.",
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
        if family == "sparse_reconstruction":
            return _cap(
                baseline,
                pde,
                task,
                sensor_mode,
                "official_adapter",
                "official",
                family,
                "Senseiver is native for sparse/irregular sensor field reconstruction with query coordinates",
                "Use official Encoder/Decoder where importable; local Perceiver variant is adapted supplement.",
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
                "coefficient/source reconstruction from sensors is a supervised Senseiver adaptation",
                "Do not include in native main table.",
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
        if family == "sparse_reconstruction":
            return _cap(
                baseline,
                pde,
                task,
                sensor_mode,
                "official_adapter",
                "official",
                family,
                "VoronoiCNN is native for sparse-sensor global field reconstruction via Voronoi tessellation and the published CNN stack",
                "If original Keras scripts are not importable, label the PyTorch Conv2D architecture reimplementation as official_architecture_reimplementation.",
                official_architecture_allowed=True,
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
                "sparse inverse is a supervised target-change adaptation, not native VoronoiCNN",
                "Supplement only.",
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
                "official_adapter",
                "canonical_math",
                family,
                "PINN is native for per-instance PDE fitting from sparse observations",
                "DeepXDE FNN may provide the architecture; the PDE objective is local and must be disclosed.",
            )
        if family == "sparse_inverse":
            if pde in STATIC_SPARSE_INVERSE_PDES:
                return _cap(
                    baseline,
                    pde,
                    task,
                    sensor_mode,
                    "official_adapter",
                    "canonical_math",
                    family,
                    "static PDE inverse has explicit solution/source or coefficient residual and sparse solution observations",
                    "PINN-style official architecture plus local PDE objective.",
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
                    "official_adapter",
                    "official",
                    family,
                    "PC-BNN sparse/noisy flow reconstruction assumptions match the 2D three-channel shallow-water field setting",
                    "Use the direct official Net when importable, or the explicit official-aligned Net/SVGD/physics-constrained reimplementation.",
                    official_aligned_allowed=True,
                    eligible_implementation_modes=("official", "official_aligned"),
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
                "Generic SVGD particles may be run only in smoke/debug as an adapted method.",
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
                "PC-BNN sparse inverse requires an explicit parameter posterior objective, not the generic field posterior here",
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
                "native",
                "canonical_math",
                family,
                "PDE-constrained optimization is a canonical per-instance sparse reconstruction baseline",
                "No single official repository is claimed.",
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
                "Supplement only; use sensor_mode=time_varying with --load-full-trajectory for main-table DA.",
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
    sensor_modes = list(sensor_modes or ["none", "random", "grid", "fixed", "time_varying"])
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
                "Report only as two_level_surrogate supplement.",
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
            "official_adapter",
            "official",
            family,
            "Senseiver supports sparse/irregular time-varying sensor reconstruction when trajectory query coordinates are supplied",
            "Use official Encoder/Decoder adapter in paper mode.",
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
            "4D-Var is canonical for time-dependent data assimilation over a trajectory",
            "Requires observation, background, and model-dynamics residual over a loaded trajectory.",
        )
    if baseline == "vivid":
        if not bool(uses_official_inverse_observation_operator):
            return _cap(
                baseline,
                pde,
                task,
                sensor_mode,
                "adapted",
                "adapted_allowed",
                family,
                "VIVID-style refinement without an official inverse-observation operator is not native VIVID",
                "Only rows that actually import/use official VIVID or invobs inverse-observation components can enter the main table.",
                eligible=False,
            )
        return _cap(
            baseline,
            pde,
            task,
            sensor_mode,
            "official_adapter",
            "official",
            family,
            "VIVID is native for learned inverse-observation initialization plus variational trajectory refinement",
            "Use direct VIVID/invobs components if importable; otherwise use the disclosed official-aligned inverse-observation plus variational refinement reimplementation.",
            official_aligned_allowed=True,
            eligible_implementation_modes=("official", "official_aligned"),
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
