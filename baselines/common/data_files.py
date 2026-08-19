from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


VALID_SPLITS = ("train", "val", "test")


def normalize_data_files(value: Any) -> dict[str, dict[str, list[str]]]:
    """Validate and normalize an explicit PDE/split file mapping."""

    if value in (None, ""):
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("data_files must be a mapping of PDE names to split file lists")

    normalized: dict[str, dict[str, list[str]]] = {}
    for raw_pde, raw_splits in value.items():
        pde = str(raw_pde).strip().lower()
        if not pde:
            raise ValueError("data_files contains an empty PDE name")
        if not isinstance(raw_splits, Mapping):
            raise ValueError(f"data_files.{pde} must be a mapping of train/val/test lists")
        unknown = sorted(set(map(str, raw_splits)) - set(VALID_SPLITS))
        if unknown:
            raise ValueError(f"data_files.{pde} contains unknown splits: {unknown}")
        split_files: dict[str, list[str]] = {}
        for split in VALID_SPLITS:
            if split not in raw_splits:
                continue
            raw_paths = raw_splits[split]
            if not isinstance(raw_paths, (list, tuple)) or not raw_paths:
                raise ValueError(f"data_files.{pde}.{split} must be a non-empty file list")
            paths = [str(path).strip() for path in raw_paths]
            if any(not path for path in paths):
                raise ValueError(f"data_files.{pde}.{split} contains an empty path")
            if len(paths) != len(set(paths)):
                raise ValueError(f"data_files.{pde}.{split} contains duplicate paths")
            split_files[split] = paths
        normalized[pde] = split_files
    return normalized


def load_data_files_from_config(
    config: Mapping[str, Any],
    *,
    repository_root: str | Path,
) -> tuple[dict[str, dict[str, list[str]]], str, str]:
    """Load inline data_files or the YAML referenced by data_files_config."""

    inline = config.get("data_files")
    reference = str(config.get("data_files_config", "") or "").strip()
    if inline not in (None, "") and reference:
        raise ValueError("configure either data_files or data_files_config, not both")
    if inline not in (None, ""):
        mapping = normalize_data_files(inline)
        return mapping, "", data_files_sha256(mapping)
    if not reference:
        return {}, "", ""

    path = Path(reference).expanduser()
    if not path.is_absolute():
        path = Path(repository_root) / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"data_files_config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if isinstance(payload, Mapping) and "data_files" in payload:
        payload = payload["data_files"]
    mapping = normalize_data_files(payload)
    return mapping, str(path), data_files_sha256(mapping)


def data_files_sha256(value: Mapping[str, Any]) -> str:
    normalized = normalize_data_files(value)
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def files_for_pde(
    mapping: Mapping[str, Mapping[str, list[str]]], pde: str
) -> dict[str, list[str]] | None:
    selected = mapping.get(str(pde).lower())
    if selected is None:
        return None
    return {split: list(paths) for split, paths in selected.items()}


def resolve_split_files(
    data_root: str | Path,
    split: str,
    data_files: Mapping[str, list[str]] | None,
) -> list[Path] | None:
    """Resolve an explicit split list below data_root without glob fallback."""

    if data_files is None:
        return None
    split = str(split).lower()
    if split not in data_files:
        raise FileNotFoundError(
            f"No explicit {split} files are configured; automatic file discovery is disabled"
        )
    root = Path(data_root).expanduser().resolve()
    resolved: list[Path] = []
    for configured in data_files[split]:
        relative = Path(configured).expanduser()
        if relative.is_absolute():
            raise ValueError(
                f"Explicit data file paths must be relative to data_root for portability: {configured}"
            )
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Explicit data file escapes data_root: {configured}") from exc
        if not path.is_file():
            raise FileNotFoundError(f"Explicitly configured data file not found: {path}")
        resolved.append(path)
    return resolved
