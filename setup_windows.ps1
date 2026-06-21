$ErrorActionPreference = "Stop"

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python launcher 'py' was not found. Install Python 3.11 first."
}

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

Write-Host "Install a matching PyTorch build first if you need NVIDIA CUDA."
Write-Host "For CPU-only experiments the next command is sufficient."
pip install -r requirements.txt
pip install -e .
python scripts/check_environment.py
