from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from .data_adapter import PDEBatch


SAMPLE_ARTIFACT_SCHEMA_VERSION = "fm4pde-evaluation-sample-v1"


class EvaluationArtifactWriter:
    """Persist every evaluated sample and render one page per sample.

    The interface deliberately accepts complete evaluation batches.  Tensor
    slicing, safe serialization, manifest integrity, and plotting stay hidden
    inside this module so the runner has one stable seam.
    """

    def __init__(
        self,
        artifact_dir: str | Path,
        *,
        run_metadata: Mapping[str, Any],
        pdf_path: str | Path | None = None,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.manifest_path = self.artifact_dir / "manifest.jsonl"
        self.pdf_path = Path(pdf_path) if pdf_path is not None else None
        self.run_metadata = _safe_mapping(run_metadata)
        self._sample_count = 0
        self._finalized = False
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text("", encoding="utf-8")

    def __enter__(self) -> "EvaluationArtifactWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            self.finalize()

    def finalize(self) -> None:
        if self._finalized:
            return
        if self.pdf_path is not None:
            render_sample_manifest_pdf(self.manifest_path, self.pdf_path)
        self._finalized = True

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "sample_artifact_count": int(self._sample_count),
            "sample_artifact_schema_version": SAMPLE_ARTIFACT_SCHEMA_VERSION,
            "sample_artifact_dir": str(self.artifact_dir),
            "sample_manifest_path": str(self.manifest_path),
            "sample_pdf_path": str(self.pdf_path) if self.pdf_path is not None else "",
        }

    def write_batch(
        self,
        batch: PDEBatch,
        prediction: torch.Tensor,
        *,
        predictive_std: torch.Tensor | None = None,
        posterior_samples: torch.Tensor | None = None,
        metrics: Sequence[Mapping[str, Any]] | None = None,
        batch_index: int,
    ) -> list[dict[str, Any]]:
        if self._finalized:
            raise RuntimeError("cannot write samples after EvaluationArtifactWriter.finalize()")
        batch_size = int(prediction.shape[0])
        if int(batch.target_fields.shape[0]) != batch_size:
            raise ValueError("prediction batch size must equal target batch size")
        if predictive_std is not None and tuple(predictive_std.shape) != tuple(prediction.shape):
            raise ValueError("predictive_std shape must equal prediction shape")
        if posterior_samples is not None and int(posterior_samples.shape[0]) != batch_size:
            raise ValueError("posterior_samples must have leading batch dimension")
        if metrics is not None and len(metrics) != batch_size:
            raise ValueError("metrics must contain one mapping per sample")

        records: list[dict[str, Any]] = []
        for item in range(batch_size):
            ordinal = self._sample_count
            artifact_path = self.artifact_dir / f"sample_{ordinal:06d}.pt"
            payload = _sample_payload(
                batch,
                prediction,
                item=item,
                ordinal=ordinal,
                batch_index=batch_index,
                run_metadata=self.run_metadata,
                predictive_std=predictive_std,
                posterior_samples=posterior_samples,
                metrics={} if metrics is None else metrics[item],
            )
            torch.save(payload, artifact_path)
            record = {
                "schema_version": SAMPLE_ARTIFACT_SCHEMA_VERSION,
                "sample_ordinal": ordinal,
                "batch_index": int(batch_index),
                "batch_item_index": item,
                "global_sample_id": payload["global_sample_id"],
                "sample_index": payload["sample_index"],
                "artifact_path": str(artifact_path.resolve()),
                "artifact_sha256": _sha256(artifact_path),
            }
            with self.manifest_path.open("a", encoding="utf-8") as manifest_handle:
                manifest_handle.write(json.dumps(record, sort_keys=True) + "\n")
            records.append(record)
            self._sample_count += 1
        return records


def load_evaluation_sample(path: str | Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    artifact_path = Path(path)
    expected = expected_sha256 or _manifest_checksum(artifact_path)
    if not expected:
        raise ValueError(f"No manifest checksum is available for evaluation sample: {artifact_path}")
    if _sha256(artifact_path) != expected:
        raise ValueError(f"Evaluation sample checksum mismatch: {artifact_path}")
    payload = torch.load(artifact_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != SAMPLE_ARTIFACT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported evaluation sample artifact: {path}")
    return payload


def render_sample_manifest_pdf(manifest_path: str | Path, output_path: str | Path) -> Path:
    manifest = Path(manifest_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    with PdfPages(output) as pdf:
        info = pdf.infodict()
        info["Title"] = "FM4PDE evaluation samples (reloaded)"
        for row in rows:
            artifact_path = Path(row["artifact_path"])
            if _sha256(artifact_path) != str(row.get("artifact_sha256", "")):
                raise ValueError(f"Evaluation sample checksum mismatch: {artifact_path}")
            figure = _sample_figure(
                load_evaluation_sample(artifact_path, expected_sha256=str(row.get("artifact_sha256", "")))
            )
            pdf.savefig(figure, bbox_inches="tight")
            figure.clear()
    return output


def _sample_payload(
    batch: PDEBatch,
    prediction: torch.Tensor,
    *,
    item: int,
    ordinal: int,
    batch_index: int,
    run_metadata: Mapping[str, Any],
    predictive_std: torch.Tensor | None,
    posterior_samples: torch.Tensor | None,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    sample_index = (
        int(batch.sample_indices.reshape(-1)[item].item())
        if isinstance(batch.sample_indices, torch.Tensor) and batch.sample_indices.numel() > item
        else item
    )
    global_id = batch.global_sample_ids[item] if item < len(batch.global_sample_ids) else str(sample_index)
    return {
        "schema_version": SAMPLE_ARTIFACT_SCHEMA_VERSION,
        "sample_ordinal": int(ordinal),
        "batch_index": int(batch_index),
        "batch_item_index": int(item),
        "global_sample_id": str(global_id),
        "sample_index": sample_index,
        "pde_name": str(batch.pde_name),
        "task": str(batch.task),
        "split": str(batch.split),
        "channel_names": list(batch.channel_names),
        "input_channel_names": list(batch.input_channel_names),
        "target_channel_names": list(batch.target_channel_names),
        "input_fields": batch.input_fields[item].detach().cpu(),
        "target_fields": batch.target_fields[item].detach().cpu(),
        "prediction": prediction[item].detach().cpu(),
        "predictive_std": (
            predictive_std[item].detach().cpu() if predictive_std is not None else torch.empty(0)
        ),
        "posterior_samples": (
            posterior_samples[item].detach().cpu() if posterior_samples is not None else torch.empty(0)
        ),
        "full_tensor": batch.full_tensor[item].detach().cpu(),
        "coords": _sample_tensor(batch.coords, item, batch.input_fields.shape[0]),
        "mask": _sample_tensor(batch.mask, item, batch.input_fields.shape[0]),
        "obs_values": _sample_tensor(batch.obs_values, item, batch.input_fields.shape[0]),
        "obs_coords": _sample_tensor(batch.obs_coords, item, batch.input_fields.shape[0]),
        "metadata": _safe_metadata(batch.metadata, item=item, batch_size=int(batch.input_fields.shape[0])),
        "pde_params": _safe_sample_mapping(batch.pde_params, item=item, batch_size=int(batch.input_fields.shape[0])),
        "metrics": _safe_mapping(metrics),
        "run_metadata": dict(run_metadata),
    }


def _sample_tensor(value: Any, item: int, batch_size: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        return torch.empty(0)
    if value.ndim > 0 and int(value.shape[0]) == int(batch_size):
        return value[item].detach().cpu()
    return value.detach().cpu()


def _safe_metadata(metadata: Mapping[str, Any], *, item: int, batch_size: int) -> dict[str, Any]:
    redundant = {
        "full_tensor",
        "full_trajectory",
        "original_input_fields",
        "observed_solution_fields",
        "observation_source_fields",
        "background_fields",
        "solution_fields",
        "source_fields",
        "coeff_fields",
        "posterior_samples",
        "predictive_std",
    }
    return _safe_sample_mapping(
        {key: value for key, value in metadata.items() if key not in redundant},
        item=item,
        batch_size=batch_size,
    )


def _safe_sample_mapping(mapping: Mapping[str, Any], *, item: int, batch_size: int) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for key, value in mapping.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and int(value.shape[0]) == batch_size:
            value = value[item]
        elif isinstance(value, (list, tuple)) and len(value) == batch_size:
            value = value[item]
        values[str(key)] = _safe_value(value)
    return values


def _safe_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _safe_value(value) for key, value in mapping.items()}


def _safe_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return _safe_mapping(value)
    if isinstance(value, (list, tuple)):
        return [_safe_value(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _sample_figure(payload: Mapping[str, Any]) -> Figure:
    target = torch.as_tensor(payload["target_fields"]).detach().cpu().float()
    prediction = torch.as_tensor(payload["prediction"]).detach().cpu().float()
    std = torch.as_tensor(payload.get("predictive_std", torch.empty(0))).detach().cpu().float()
    channels = max(int(target.shape[0]) if target.ndim >= 2 else 1, 1)
    figure = Figure(figsize=(15, max(3.2, 3.0 * channels)), constrained_layout=True)
    axes = figure.subplots(channels, 4, squeeze=False)
    names = list(payload.get("target_channel_names", []))
    for channel in range(channels):
        target_field = target[channel] if target.ndim >= 2 else target
        pred_field = prediction[channel] if prediction.ndim >= 2 else prediction
        error = (pred_field - target_field).abs()
        std_field = std[channel] if std.numel() and std.ndim >= 2 else torch.zeros_like(target_field)
        channel_name = names[channel] if channel < len(names) else f"channel {channel}"
        for axis, field, title in zip(
            axes[channel],
            (target_field, pred_field, error, std_field),
            ("target", "prediction", "absolute error", "predictive std"),
        ):
            _draw_field(axis, field)
            axis.set_title(f"{channel_name}: {title}")
    metrics = payload.get("metrics", {})
    metric_text = ", ".join(
        f"{key}={value:.4g}" if isinstance(value, (int, float)) else f"{key}={value}"
        for key, value in metrics.items()
    )
    figure.suptitle(
        f"{payload.get('pde_name', '')} | {payload.get('task', '')} | sample={payload.get('global_sample_id', '')}"
        + (f"\n{metric_text}" if metric_text else "")
    )
    return figure


def _draw_field(axis, field: torch.Tensor) -> None:
    array = field.detach().cpu().numpy().squeeze()
    if array.ndim <= 1:
        axis.plot(array.reshape(-1))
        axis.grid(alpha=0.2)
        return
    while array.ndim > 2:
        array = array[0]
    image = axis.imshow(array, origin="lower", aspect="auto", cmap="viridis")
    axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_checksum(artifact_path: Path) -> str:
    manifest = artifact_path.parent / "manifest.jsonl"
    if not manifest.exists():
        return ""
    resolved = artifact_path.resolve()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if Path(row.get("artifact_path", "")).resolve() == resolved:
            return str(row.get("artifact_sha256", ""))
    return ""
