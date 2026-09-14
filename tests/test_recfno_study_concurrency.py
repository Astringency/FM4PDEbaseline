import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_identical_concurrent_training_requests_only_train_once(tmp_path):
    command = [sys.executable, str(ROOT / "scripts/run_recfno_variable_sensors.py"),
               "--train-sensor-counts", "2,5", "--baseline", "recfno", "--pde", "poisson",
               "--task", "sparse_solution_multicondition", "--num-sensors", "5",
               "--sensor-mode", "random_per_sample", "--sensor-budget-mode", "total",
               "--train-size", "12", "--val-size", "4", "--test-size", "2", "--batch-size", "2",
               "--epochs", "1", "--synthetic-data", "--synthetic-resolution", "8", "--num-workers", "0",
               "--implementation-mode", "adapted", "--official-backend", "local", "--train-only",
               "--output-dir", str(tmp_path / "run"), "--run-id", "concurrency",
               "--method-override", "width=4", "--method-override", "modes1=2", "--method-override", "modes2=2",
               "--method-override", "normalize=true"]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    logs, processes = [], []
    try:
        for index in range(2):
            handle = (tmp_path / f"process{index}.log").open("w")
            logs.append(handle)
            processes.append(subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT))
        for process in processes:
            assert process.wait(timeout=60) == 0
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
        for handle in logs:
            handle.close()
    text = "\n".join((tmp_path / f"process{i}.log").read_text() for i in range(2))
    assert text.count("[run stage] fit start") == 1
    assert text.count("reuse checksum-validated completed result") == 1
    history = json.loads((tmp_path / "run/concurrency_train_history.json").read_text())
    assert history["completed_epochs"] == 1
    audit = (tmp_path / "run/sensor_count_audit.jsonl").read_text().splitlines()
    assert len(audit) == 2  # Exactly one normalization pass and one training epoch.
