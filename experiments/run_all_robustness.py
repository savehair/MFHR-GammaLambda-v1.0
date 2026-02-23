from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List


def run_script(script_path: str) -> Dict[str, float | int | str]:
    start = time.time()
    proc = subprocess.run(
        [sys.executable, script_path],
        check=False,
        capture_output=True,
        text=True,
    )
    elapsed = float(time.time() - start)

    if proc.returncode != 0:
        raise RuntimeError(
            f"Script failed: {script_path}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )

    return {
        "script": script_path,
        "returncode": int(proc.returncode),
        "elapsed_sec": elapsed,
    }


def main() -> None:
    scripts: List[str] = [
        "experiments/robustness/nif_extreme_flow_test.py",
        "experiments/robustness/nif_ood_test.py",
        "experiments/robustness/model_comparison.py",
        "experiments/robustness/history_sensitivity.py",
        "experiments/robustness/scale_expansion_test.py",
        "experiments/robustness/peak_shock_test.py",
        "experiments/robustness/boundary_scan.py",
    ]

    records: List[Dict[str, float | int | str]] = []
    for script in scripts:
        records.append(run_script(script))

    Path("results").mkdir(exist_ok=True)
    summary_path = Path("results/robustness_summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


# pytest assertion example
def test_run_all_script_order_example() -> None:
    scripts = [
        "experiments/robustness/nif_extreme_flow_test.py",
        "experiments/robustness/nif_ood_test.py",
        "experiments/robustness/model_comparison.py",
        "experiments/robustness/history_sensitivity.py",
        "experiments/robustness/scale_expansion_test.py",
        "experiments/robustness/peak_shock_test.py",
        "experiments/robustness/boundary_scan.py",
    ]
    assert scripts[0].endswith("nif_extreme_flow_test.py")
    assert scripts[-1].endswith("boundary_scan.py")


if __name__ == "__main__":
    main()
