import importlib.util
import sys
from pathlib import Path
from typing import Any


class ModelRuntimeError(RuntimeError):
    """Raised when a supplied model handoff cannot provide the required output."""


def load_handoff_module(module_name: str, handoff_dir: Path) -> Any:
    module_path = handoff_dir / "inference.py"
    if not module_path.exists():
        raise ModelRuntimeError(f"Inference module missing: {module_path}")

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ModelRuntimeError(f"Could not load inference module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(handoff_dir))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def load_model_runtimes(binary_dir: Path, segmentation_dir: Path) -> tuple[Any, Any, dict[str, Any]]:
    binary_module = load_handoff_module("oilguard_binary_inference", binary_dir)
    segmentation_module = load_handoff_module("oilguard_segmentation_inference", segmentation_dir)

    binary_ready = callable(getattr(binary_module, "predict", None))
    segmentation_predict = getattr(segmentation_module, "predict", None)
    segmentation_ready = callable(segmentation_predict)

    if segmentation_ready:
        sample = segmentation_dir / "sample_images" / "sample_oil_1.jpg"
        if sample.exists():
            result = segmentation_predict(sample)
            segmentation_ready = isinstance(result, dict) and (
                "mask" in result or "mask_path" in result or "segmentation_mask" in result
            )

    status = {
        "binary": "ready" if binary_ready else "error",
        "segmentation": "ready" if segmentation_ready else "error",
        "segmentation_message": None if segmentation_ready else (
            "The supplied segmentation handoff returns classification fields, "
            "not a segmentation mask. A real segmentation checkpoint/inference contract is required."
        ),
    }
    return binary_module, segmentation_module, status
