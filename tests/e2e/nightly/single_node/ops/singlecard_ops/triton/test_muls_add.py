import pytest
import torch

from tests.accuracy import AccuracyTolerance, assert_close
from vllm_ascend.ops.triton.muls_add import muls_add_triton
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

MULS_ADD_TOLERANCE = AccuracyTolerance(rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    ("shape", "dtype", "scale"),
    [
        ((1, 2048), torch.float16, 1.25),
        ((4000, 2048), torch.float16, 0.75),
        ((4, 2048), torch.bfloat16, 1.0),
    ],
)
@torch.inference_mode()
def test_muls_add_triton_correctness(shape, dtype, scale):
    """compare the correctness of muls_add_triton with the PyTorch baseline implementation."""
    init_device_properties_triton()
    device = "npu"

    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=dtype, device=device)
    y = torch.randn(*shape, dtype=dtype, device=device)

    out_triton = muls_add_triton(x, y, scale)
    out_ref = x * scale + y

    assert out_triton.shape == out_ref.shape
    assert out_triton.dtype == out_ref.dtype
    assert_close(
        out_triton,
        out_ref,
        tolerance=MULS_ADD_TOLERANCE,
        name=f"muls_add.{dtype}",
        reason="preserve the elementwise multiply-add kernel's validated low-precision error bound",
    )
