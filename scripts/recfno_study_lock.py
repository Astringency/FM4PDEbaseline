"""Serialize duplicate requests for one study output directory."""
from contextlib import contextmanager
import fcntl
from pathlib import Path


@contextmanager
def output_lock(directory: Path, phase: str):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f".{phase}.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"[{phase}] waiting for the active owner of {directory}", flush=True)
            fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
