from __future__ import annotations

import importlib
import importlib.util
import io
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_ROOT = ROOT / "offical"


OFFICIAL_SOURCE_INFO: dict[str, dict[str, str]] = {
    "neuraloperator": {
        "official_repo": "https://github.com/neuraloperator/neuraloperator",
        "official_commit_or_version": "vendored",
        "official_import_path": str(OFFICIAL_ROOT / "neuraloperator" / "neuralop" / "models"),
    },
    "deepxde": {
        "official_repo": "https://github.com/lululxvi/deepxde",
        "official_commit_or_version": "vendored",
        "official_import_path": str(OFFICIAL_ROOT / "deepxde" / "deepxde"),
    },
    "recfno": {
        "official_repo": "https://github.com/zhaoxiaoyu1995/recfno",
        "official_commit_or_version": "vendored",
        "official_import_path": str(OFFICIAL_ROOT / "RecFNO" / "model"),
    },
    "senseiver": {
        "official_repo": "https://github.com/OrchardLANL/Senseiver",
        "official_commit_or_version": "vendored",
        "official_import_path": str(OFFICIAL_ROOT / "Senseiver" / "model.py"),
    },
    "pc_bnn": {
        "official_repo": "https://github.com/Jianxun-Wang/Physics-constrained-Bayesian-deep-learning",
        "official_commit_or_version": "vendored",
        "official_import_path": str(OFFICIAL_ROOT / "PC-BNN" / "code"),
    },
    "ifno": {
        "official_repo": "https://github.com/BayesianAIGroup/iFNO",
        "official_commit_or_version": "vendored",
        "official_import_path": str(OFFICIAL_ROOT / "iFNO"),
    },
    "vivid": {
        "official_repo": "https://github.com/DL-WG/VIVID",
        "official_commit_or_version": "vendored",
        "official_import_path": str(OFFICIAL_ROOT / "VIVID"),
    },
    "invobs": {
        "official_repo": "https://github.com/googleinterns/invobs-data-assimilation",
        "official_commit_or_version": "vendored",
        "official_import_path": str(OFFICIAL_ROOT / "invobs-data-assimilation"),
    },
    "vivid_invobs": {
        "official_repo": "https://github.com/DL-WG/VIVID; https://github.com/googleinterns/invobs-data-assimilation",
        "official_commit_or_version": "vendored",
        "official_import_path": f"{OFFICIAL_ROOT / 'VIVID'}; {OFFICIAL_ROOT / 'invobs-data-assimilation'}",
    },
    "voronoi_cnn": {
        "official_repo": "https://github.com/kfukami/Voronoi-CNN",
        "official_commit_or_version": "vendored",
        "official_import_path": str(OFFICIAL_ROOT / "Voronoi-CNN"),
    },
}


class OfficialImportError(ImportError):
    """Raised when a vendored official implementation is unavailable."""


def official_source_info(source: str) -> dict[str, str]:
    return dict(OFFICIAL_SOURCE_INFO.get(str(source), {}))


def requested_implementation_mode(config: dict[str, Any]) -> str:
    if "implementation_mode" in config:
        return str(config.get("implementation_mode") or "").lower()
    backend = str(config.get("official_backend", "auto")).lower()
    if backend in {"local", "none", "adapted"}:
        return "adapted"
    if backend in {"official", "neuraloperator", "deepxde", "recfno", "senseiver", "pc_bnn", "ifno"}:
        return "official"
    return backend or "auto"


def allows_adapted_fallback(config: dict[str, Any]) -> bool:
    mode = requested_implementation_mode(config)
    if mode in {"official", "official_or_skip", "canonical_math", "official_architecture", "official_aligned"}:
        return False
    backend = str(config.get("official_backend", "auto")).lower()
    return backend not in {"official"}


@contextmanager
def _prepend_path(path: Path):
    text = str(path)
    inserted = False
    if text not in sys.path:
        sys.path.insert(0, text)
        inserted = True
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(text)
            except ValueError:
                pass


def _load_file_module(name: str, path: Path) -> ModuleType:
    if not path.exists():
        raise OfficialImportError(f"Official module file not found: {path}")
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise OfficialImportError(f"Cannot load official module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(name, None)
        raise OfficialImportError(str(exc)) from exc
    return module


def get_recfno_fno_classes() -> tuple[type[Any], type[Any], type[Any]]:
    module = _load_file_module("_fm4pde_official_recfno_fno", OFFICIAL_ROOT / "RecFNO" / "model" / "fno.py")
    return module.FNO2d, module.VoronoiFNO2d, module.SpectralConv2d


def get_recfno_unet_class() -> type[Any]:
    module = _load_file_module("_fm4pde_official_recfno_cnn", OFFICIAL_ROOT / "RecFNO" / "model" / "cnn.py")
    return module.UNet


def get_neuraloperator_fno_class() -> type[Any]:
    with _prepend_path(OFFICIAL_ROOT / "neuraloperator"):
        try:
            module = importlib.import_module("neuralop.models")
            return module.FNO
        except Exception as exc:
            raise OfficialImportError(str(exc)) from exc


def get_deepxde_deeponet_class() -> type[Any]:
    return _deepxde_import("deepxde.nn.pytorch.deeponet", "DeepONetCartesianProd")


def get_deepxde_fnn_class() -> type[Any]:
    return _deepxde_import("deepxde.nn.pytorch.fnn", "FNN")


def _deepxde_import(module_name: str, attr: str) -> type[Any]:
    default_device = None
    try:
        import torch

        default_device = torch.get_default_device()
    except Exception:
        torch = None  # type: ignore[assignment]
    with _prepend_path(OFFICIAL_ROOT / "deepxde"):
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                module = importlib.import_module(module_name)
            return getattr(module, attr)
        except Exception as exc:
            raise OfficialImportError(str(exc)) from exc
        finally:
            if default_device is not None and torch is not None:
                torch.set_default_device(default_device)


def get_senseiver_classes() -> tuple[type[Any], type[Any]]:
    module = _load_file_module("_fm4pde_official_senseiver_model", OFFICIAL_ROOT / "Senseiver" / "model.py")
    return module.Encoder, module.Decoder


def get_pc_bnn_net_class() -> type[Any]:
    with _prepend_path(OFFICIAL_ROOT / "PC-BNN" / "code"):
        try:
            module = importlib.import_module("FCN")
            return module.Net
        except Exception as exc:
            raise OfficialImportError(str(exc)) from exc


def get_ifno_official_status() -> None:
    """Validate whether vendored iFNO can be imported as a library component.

    The current upstream snapshot is organized as executable scripts that parse
    command-line globals at import time. The wrapper therefore refuses to claim
    official code reuse until a stable importable module adapter is added.
    """

    path = OFFICIAL_ROOT / "iFNO"
    if not path.exists():
        raise OfficialImportError(f"iFNO source tree not found: {path}")
    raise OfficialImportError("vendored iFNO scripts are not safely importable model components")


def get_ifno_official_aligned_status() -> None:
    """Validate local availability of the iFNO official-aligned adapter."""

    path = OFFICIAL_ROOT / "iFNO"
    if not path.exists():
        raise OfficialImportError(f"iFNO source tree not found: {path}")
    try:
        import baselines.methods.ifno_official_aligned  # noqa: F401
    except Exception as exc:
        raise OfficialImportError(f"iFNO official-aligned adapter unavailable: {exc}") from exc


def get_vivid_official_status() -> None:
    """Validate whether vendored VIVID/invobs can be used as official components.

    The vendored VIVID snapshot is organized as executable Keras scripts and the
    invobs repository needs a dataset-specific training/config pipeline. Until a
    stable importable inverse-observation adapter is added, the local refinement
    path must remain VIVID-style supplement or be skipped in paper official mode.
    """

    vivid = OFFICIAL_ROOT / "VIVID"
    invobs = OFFICIAL_ROOT / "invobs-data-assimilation"
    if not vivid.exists():
        raise OfficialImportError(f"VIVID source tree not found: {vivid}")
    if not invobs.exists():
        raise OfficialImportError(f"invobs source tree not found: {invobs}")
    raise OfficialImportError("vendored VIVID/invobs is not exposed as a stable importable inverse-observation adapter")


def get_vivid_official_aligned_status() -> None:
    """Validate local availability of the VIVID/invobs official-aligned adapter."""

    vivid = OFFICIAL_ROOT / "VIVID"
    invobs = OFFICIAL_ROOT / "invobs-data-assimilation"
    if not vivid.exists():
        raise OfficialImportError(f"VIVID source tree not found: {vivid}")
    if not invobs.exists():
        raise OfficialImportError(f"invobs source tree not found: {invobs}")
    try:
        import baselines.methods.vivid_official_aligned  # noqa: F401
    except Exception as exc:
        raise OfficialImportError(f"VIVID official-aligned adapter unavailable: {exc}") from exc


def get_pc_bnn_official_aligned_status() -> None:
    """Validate local availability of the PC-BNN official-aligned adapter."""

    path = OFFICIAL_ROOT / "PC-BNN" / "code"
    if not path.exists():
        raise OfficialImportError(f"PC-BNN source tree not found: {path}")
    try:
        import baselines.methods.pc_bnn  # noqa: F401
    except Exception as exc:
        raise OfficialImportError(f"PC-BNN official-aligned adapter unavailable: {exc}") from exc
