#!/usr/bin/env bash
# Cloud Agent install script for the Intelligent Video platform.
# Idempotent: safe to run repeatedly and against a warm/snapshotted VM.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# System packages
#   - python3.12-venv: create the backend virtualenv (ensurepip)
#   - ffmpeg: video probing/transcoding used by the upload pipeline
# opencv-python-headless needs no extra GUI libs.
# ---------------------------------------------------------------------------
need_apt=0
dpkg -s python3.12-venv >/dev/null 2>&1 || need_apt=1
command -v ffmpeg >/dev/null 2>&1 || need_apt=1
if [ "$need_apt" -eq 1 ]; then
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends python3.12-venv ffmpeg
fi

# ---------------------------------------------------------------------------
# Backend: Python virtualenv + dependencies (CPU wheels; no GPU on the VM)
# ---------------------------------------------------------------------------
python3 -m venv backend/venv
backend/venv/bin/pip install --upgrade pip
backend/venv/bin/pip install -r backend/requirements.txt
# Dev tooling documented in CONTRIBUTING.md (linter + test runner).
backend/venv/bin/pip install ruff pytest

# ---------------------------------------------------------------------------
# Frontend: Node dependencies
# ---------------------------------------------------------------------------
(cd frontend && npm ci)

# ---------------------------------------------------------------------------
# Pre-cache the OWLv2 model weights so the backend starts fast on first boot
# and does not depend on network access at runtime. Cached under
# ~/.cache/huggingface, which is captured in the environment snapshot.
# ---------------------------------------------------------------------------
backend/venv/bin/python - <<'PY'
from transformers import Owlv2ForObjectDetection, Owlv2Processor

model = "google/owlv2-base-patch16-ensemble"
Owlv2Processor.from_pretrained(model)
Owlv2ForObjectDetection.from_pretrained(model)
print("OWLv2 model cached")
PY

echo "install.sh complete"
