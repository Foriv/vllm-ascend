import gc

import pytest
import torch
import torch.nn.functional as F

from tests.accuracy import AccuracyTolerance, assert_close
from vllm_ascend.ops.triton.fla.l2norm import l2norm_fwd
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


def _l2norm_tolerance(dtype: torch.dtype) -> AccuracyTolerance:
    if dtype == torch.float16:
        return AccuracyTolerance(rtol=3e-3, atol=5e-3)
    if dtype == torch.bfloat16:
        return AccuracyTolerance(rtol=1e-2, atol=5e-2)
    if dtype == torch.float32:
        return AccuracyTolerance(rtol=3e-4, atol=1e-3)
    raise ValueError(f"Unsupported L2Norm dtype: {dtype}")


@pytest.mark.parametrize(
    ("B", "T", "H", "D", "dtype"),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-{}".format(*test))
        for test in [
            (1, 63, 1, 60, torch.float),
            (2, 500, 4, 64, torch.float),
            (2, 1000, 2, 100, torch.float),
            (3, 1024, 4, 128, torch.float),
        ]
    ],
)
def test_l2norm(B: int, T: int, H: int, D: int, dtype: torch.dtype):
    torch.manual_seed(42)
    init_device_properties_triton()
    device = "npu"
    x = torch.randn(B, T, H, D, dtype=dtype).to(device).requires_grad_(True)
    x = x * 0.5 + 0.3

    ref = F.normalize(x, dim=-1, p=2)
    tri = l2norm_fwd(x)

    assert_close(
        tri,
        ref,
        tolerance=_l2norm_tolerance(dtype),
        name="l2norm_fwd",
        reason="preserves the existing Ascend L2Norm bounds",
    )
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
