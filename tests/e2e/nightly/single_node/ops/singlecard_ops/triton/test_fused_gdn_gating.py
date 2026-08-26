import torch

from tests.accuracy import AccuracyTolerance, assert_close
from vllm_ascend._310p.ops.fla.fused_gdn_gating import fused_gdn_gating_pytorch
from vllm_ascend.ops.triton.fused_gdn_gating import fused_gdn_gating_patch
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


_FUSED_GDN_GATING_TOLERANCE = AccuracyTolerance(rtol=1e-2, atol=1e-2)


def test_fused_gdn_gating_310p_parity_precision():
    init_device_properties_triton()
    torch.manual_seed(0)
    device = "npu"

    num_tokens = 37
    num_heads = 8

    A_log = torch.randn(num_heads, dtype=torch.float16, device=device)
    dt_bias = torch.randn(num_heads, dtype=torch.float16, device=device)
    a = torch.randn(num_tokens, num_heads, dtype=torch.float16, device=device)
    b = torch.randn(num_tokens, num_heads, dtype=torch.float16, device=device)

    triton_g, triton_beta = fused_gdn_gating_patch(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        beta=1.0,
        threshold=20.0,
    )
    ref_g, ref_beta = fused_gdn_gating_pytorch(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        beta=1.0,
        threshold=20.0,
    )

    assert_close(
        triton_g.to(torch.float32).cpu(),
        ref_g.to(torch.float32).cpu(),
        policy_dtype=triton_g.dtype,
        tolerance=_FUSED_GDN_GATING_TOLERANCE,
        equal_nan=True,
        name="fused_gdn_gating_g",
        reason="preserves the existing FP16 gating parity bound",
    )
    assert_close(
        triton_beta.to(torch.float32).cpu(),
        ref_beta.to(torch.float32).cpu(),
        policy_dtype=triton_beta.dtype,
        tolerance=_FUSED_GDN_GATING_TOLERANCE,
        equal_nan=True,
        name="fused_gdn_gating_beta",
        reason="preserves the existing FP16 gating parity bound",
    )
