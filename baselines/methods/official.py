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


class OfficialImportError(ImportError):
    """Raised when a vendored official implementation is unavailable."""


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
