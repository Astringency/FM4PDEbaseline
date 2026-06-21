from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_experiment_shell_scripts_pass_bash_syntax_check():
    scripts = sorted(str(path.relative_to(ROOT)) for path in (ROOT / "scripts/experiments").glob("*.sh"))
    subprocess.run(["bash", "-n", *scripts], cwd=ROOT, check=True)
