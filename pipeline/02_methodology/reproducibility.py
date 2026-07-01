"""Generator REPRODUCIBILITY.md — wersje bibliotek, git hash, hardware, data hash.

Wymóg konkursowy: każdy run musi być reproducible.

Użycie:
    python reproducibility.py --output pipeline/06_thesis_inputs/REPRODUCIBILITY.md
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("reproducibility")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _git_branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _git_status() -> str:
    try:
        return subprocess.check_output(
            ["git", "status", "--short"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _pip_freeze() -> str:
    try:
        return subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], stderr=subprocess.DEVNULL
        ).decode()
    except Exception:
        return ""


def _data_manifest_hash() -> str:
    try:
        manifest_json = Path("datasets/MANIFEST.json")
        if manifest_json.exists():
            import hashlib
            h = hashlib.sha256()
            h.update(manifest_json.read_bytes())
            return h.hexdigest()[:16]
    except Exception:
        pass
    return "unknown"


def _hardware_info() -> dict[str, str]:
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "?",
        "python_version": sys.version.split()[0],
    }
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = str(torch.cuda.is_available())
        info["mps_available"] = str(torch.backends.mps.is_available())
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch_version"] = "not installed"

    # RAM
    try:
        import psutil
        info["ram_gb"] = f"{psutil.virtual_memory().total / 1024**3:.1f}"
    except ImportError:
        pass
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="pipeline/06_thesis_inputs/REPRODUCIBILITY.md")
    parser.add_argument("--pip-freeze-output", default="pipeline/06_thesis_inputs/pip_freeze.txt")
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    hw = _hardware_info()
    git_h = _git_hash()
    git_b = _git_branch()
    git_s = _git_status()
    data_h = _data_manifest_hash()
    pip = _pip_freeze()

    lines = [
        "# Reproducibility manifest",
        "",
        f"Generated: {datetime.utcnow().isoformat()}Z",
        "",
        "## Git",
        "",
        f"- Branch: `{git_b}`",
        f"- Commit: `{git_h}`",
        f"- Dirty working tree: {'YES (see below)' if git_s else 'NO'}",
        "",
    ]
    if git_s:
        lines.append("```")
        lines.append(git_s)
        lines.append("```")
        lines.append("")

    lines.extend([
        "## Hardware",
        "",
    ])
    for k, v in hw.items():
        lines.append(f"- {k}: {v}")
    lines.append("")

    lines.extend([
        "## Data",
        "",
        f"- `datasets/MANIFEST.json` SHA-256 (first 16): `{data_h}`",
        "",
        "Pełny manifest plików: `datasets/MANIFEST.md`",
        "",
        "## Python environment",
        "",
        f"Pełny `pip freeze` zapisany w `{args.pip_freeze_output}`.",
        "",
        "Kluczowe wersje:",
    ])

    # Wyciągnij kluczowe pakiety z pip freeze
    key_packages = ["torch", "transformers", "peft", "mlx-lm", "scikit-learn",
                    "xgboost", "datasketch", "trafilatura", "modal"]
    for line in pip.splitlines():
        for pkg in key_packages:
            if line.startswith(pkg + "=") or line.startswith(pkg + " "):
                lines.append(f"- `{line.strip()}`")
                break

    out.write_text("\n".join(lines), encoding="utf-8")
    Path(args.pip_freeze_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.pip_freeze_output).write_text(pip, encoding="utf-8")
    logger.info("Reproducibility → %s", out)
    logger.info("Pip freeze → %s", args.pip_freeze_output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
