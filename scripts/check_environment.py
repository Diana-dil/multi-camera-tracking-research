from __future__ import annotations

import importlib
import platform
import sys


REQUIRED_MODULES = [
    "torch",
    "ultralytics",
    "cv2",
    "numpy",
    "pandas",
    "yaml",
    "scipy",
]


def main() -> None:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    failures: list[str] = []

    for module_name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "unknown")
            print(f"[OK] {module_name}: {version}")
        except Exception as exc:  # environment diagnostics should report all failures
            failures.append(module_name)
            print(f"[FAIL] {module_name}: {exc}")

    try:
        import torch

        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except Exception:
        pass

    if failures:
        raise SystemExit(
            "Missing or broken modules: " + ", ".join(failures) +
            ". Recreate the virtual environment and reinstall requirements."
        )


if __name__ == "__main__":
    main()
