"""Phase 1 smoke tests: verify the pinned SD stack is importable and mechanically
correct, without requiring a full/expensive real-checkpoint generation.

Run with:  .venv/Scripts/python.exe -m pytest tests/test_environment.py -v
"""
import numpy as np
import pytest


def test_core_imports():
    import torch  # noqa: F401
    import torchvision  # noqa: F401
    import diffusers  # noqa: F401
    import transformers  # noqa: F401
    import accelerate  # noqa: F401
    import PIL  # noqa: F401


def test_pinned_versions():
    import diffusers
    import transformers
    import PIL

    assert diffusers.__version__ == "0.24.0"
    assert transformers.__version__ == "4.37.2"
    # PRISM pins pillow==9.0.1, but that conflicts with scikit-image>=0.25 (needs
    # pillow>=10.1); we relaxed this pin, see docs/PHASE_1_AUDIT.md section F.
    from packaging.version import Version

    assert Version(PIL.__version__) >= Version("10.1")


def test_no_cuda_available():
    """Documents the hardware constraint this project runs under (see docs/PHASE_1_AUDIT.md).
    This is expected to be True on the development machine (Intel iGPU, no CUDA)."""
    import torch

    assert torch.cuda.is_available() is False


def test_tiny_pipeline_end_to_end():
    """Proves the full SD call path (tokenize -> text-encode -> denoise loop ->
    VAE decode -> PIL image) works with our pinned stack, using a tiny dummy-weight
    model so the test stays fast and needs no large download."""
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        "hf-internal-testing/tiny-stable-diffusion-pipe", safety_checker=None
    )
    pipe.to("cpu")
    out = pipe("chest x-ray of a patient", num_inference_steps=2)
    img = out.images[0]

    arr = np.array(img)
    assert img.mode == "RGB"
    assert arr.shape[0] > 0 and arr.shape[1] > 0
    assert not np.isnan(arr.astype(float)).any()
    assert arr.min() >= 0 and arr.max() <= 255


def test_prism_editing_pipeline_import_fails_as_expected():
    """Documents the known blocker (docs/PHASE_1_AUDIT.md section A): the official
    PRISM repo is missing utils/misc_utils.py, utils/edit_utils.py,
    utils/inference_utils.py and generate_captions.py. This test pins down that the
    failure is exactly this ModuleNotFoundError, so a future re-run of this suite
    will flag it loudly if the upstream repo is ever fixed (i.e. this test should
    eventually start FAILING, which is the signal to remove it)."""
    import sys
    from pathlib import Path

    prism_path = Path(__file__).resolve().parents[1] / "external" / "PRISM"
    if not prism_path.exists():
        pytest.skip("external/PRISM not cloned in this checkout")

    sys.path.insert(0, str(prism_path))
    try:
        with pytest.raises(ModuleNotFoundError, match="generate_captions"):
            import generate_cf_images  # noqa: F401
    finally:
        sys.path.remove(str(prism_path))
        sys.modules.pop("generate_cf_images", None)
