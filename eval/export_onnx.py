"""Export the pose model to ONNX, and verify the export is faithful.

Why this exists: the ~2 GB `torch`/`ultralytics` install is by far the biggest
thing standing between this project and a real free-tier deployment (see
docs/ARCHITECTURE.md's "Scaling beyond one process" and docs/ROADMAP.md). ONNX
is the step toward that — but it is two separable claims, and this module only
makes one of them:

  1. The exported graph computes the same function as the trained weights.
     This is what `verify_export` checks, and what this module delivers.
  2. The live judge can run inference *without installing torch at all*.
     `ultralytics` hard-imports `torch` even to run an `.onnx` file through its
     own `YOLO(...)` API (confirmed by reading `ultralytics/engine/predictor.py`
     and `ultralytics/nn/autobackend.py` — both import torch unconditionally,
     regardless of which backend actually executes the graph). Realising claim
     2 means bypassing ultralytics' Python API for the live path entirely:
     onnxruntime directly, with the letterbox resize, keypoint decode, and NMS
     ultralytics currently does for us reimplemented by hand. That is a rewrite
     of `whitelights/pose.py` — the one fully-implemented, load-bearing module
     in this codebase — and it deserves real footage to validate against
     before it replaces the tested path. Deliberately not attempted here.

Sizes, measured on this machine (macOS arm64, CPU-only wheels — Linux/CUDA
wheels used by the Dockerfile are considerably larger):

    torch + torchvision installed   ~530 MB
    onnxruntime installed            ~77 MB
    yolo11n-pose.pt                  ~6 MB
    yolo11n-pose.onnx                ~11 MB

The model weights were never the ~2 GB — that number is the dependency
footprint (torch itself, plus CUDA-adjacent packages on Linux). The win from
ONNX is dropping that dependency at deploy time, not a smaller weight file;
the exported graph is in fact slightly larger on disk than the original.

Needs the ``cv`` and ``onnx`` extras::

    pip install -e ".[cv,onnx]"
    python -m eval.export_onnx --model yolo11n-pose.pt --verify
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Pure helpers (no ultralytics / torch / onnxruntime — unit-tested directly)
# ---------------------------------------------------------------------------


def format_size(num_bytes: int) -> str:
    """Human-readable file size, e.g. ``11.3 MB``."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024
    return f"{size:.1f} GB"  # pragma: no cover - unreachable, satisfies mypy


@dataclass
class ExportVerification:
    """Result of comparing raw model outputs between the `.pt` and `.onnx` graphs.

    Compares *raw* forward-pass output (before NMS / keypoint decode), on
    identical random input, so this checks graph fidelity independent of
    whether any particular image contains a detectable person — the property
    under test is "did the export preserve the trained weights faithfully",
    not "can this model find a lifter in this frame".
    """

    max_abs_diff: float
    mean_abs_diff: float
    output_shape: tuple[int, ...]
    atol: float

    @property
    def passed(self) -> bool:
        return self.max_abs_diff <= self.atol

    def __str__(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        return (
            f"{verdict}  max_abs_diff={self.max_abs_diff:.6f}  "
            f"mean_abs_diff={self.mean_abs_diff:.6f}  "
            f"(tolerance {self.atol}, output shape {self.output_shape})"
        )


# ---------------------------------------------------------------------------
# Export + verification (needs the `cv` and `onnx` extras)
# ---------------------------------------------------------------------------


def export_to_onnx(model_path: str | Path, *, imgsz: int = 640) -> Path:
    """Export a `.pt` pose model to ONNX. Returns the output path.

    Thin wrapper around ultralytics' own exporter (opset selection, graph
    simplification via onnxslim, and the `.onnx` writer are all theirs) —
    reimplementing that would be needless risk for zero benefit, unlike the
    live inference path this module deliberately does not touch.
    """
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    exported = model.export(format="onnx", imgsz=imgsz, simplify=True)
    return Path(exported)


def verify_export(
    pt_path: str | Path,
    onnx_path: str | Path,
    *,
    imgsz: int = 640,
    seed: int = 0,
    atol: float = 0.01,
) -> ExportVerification:
    """Run identical random input through both graphs and compare raw output.

    ``atol`` defaults to 0.01 against an output range that typically spans
    several hundred (pixel-scale box/keypoint coordinates) — loose enough to
    absorb ordinary floating-point and op-implementation differences between
    torch and onnxruntime, tight enough to catch an export that silently
    dropped or misdecoded part of the graph.
    """
    from typing import Any

    import numpy as np
    import onnxruntime as ort
    import torch
    from ultralytics import YOLO

    torch.manual_seed(seed)
    x = torch.rand(1, 3, imgsz, imgsz)

    # Typed Any to match whitelights/pose.py's convention for ultralytics
    # objects: real stubs (when the `cv` extra happens to be installed
    # locally) and `ignore_missing_imports` (CI, which never installs it) type
    # this attribute differently, and neither is worth fighting — `.model` is
    # a torch.nn.Module once YOLO(...) has loaded a real checkpoint.
    torch_model: Any = YOLO(str(pt_path)).model
    torch_model.eval()
    with torch.no_grad():
        torch_out = torch_model(x)
    if isinstance(torch_out, (tuple, list)):
        torch_out = torch_out[0]
    torch_out = torch_out.numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    onnx_out = session.run(None, {input_name: x.numpy()})[0]

    diff = np.abs(torch_out - onnx_out)
    return ExportVerification(
        max_abs_diff=float(diff.max()),
        mean_abs_diff=float(diff.mean()),
        output_shape=tuple(torch_out.shape),
        atol=atol,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, default=Path("yolo11n-pose.pt"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--verify", action="store_true", help="compare raw outputs against the .pt model"
    )
    parser.add_argument("--atol", type=float, default=0.01)
    args = parser.parse_args()

    onnx_path = export_to_onnx(args.model, imgsz=args.imgsz)

    pt_size = args.model.stat().st_size
    onnx_size = onnx_path.stat().st_size
    print(f"\n{args.model.name}: {format_size(pt_size)}")
    print(f"{onnx_path.name}: {format_size(onnx_size)}")

    if args.verify:
        result = verify_export(args.model, onnx_path, imgsz=args.imgsz, atol=args.atol)
        print(f"\n{result}")
        if not result.passed:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
