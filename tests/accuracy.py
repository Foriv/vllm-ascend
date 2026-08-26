# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Shared numerical-accuracy policies for operator tests.

The reference triton-ascend-kernels repository combines PyTorch's dtype
defaults with operator-specific bounds. This module turns that approach into a
backend-independent vllm-ascend test policy: use the dtype baseline by default,
exact comparison for discrete contracts, documented overrides for numerical
exceptions, and a separate NRMSE metric for recurrent accumulation.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch

_NRMSE_DENOMINATOR_EPSILON = 1e-8


@dataclass(frozen=True)
class AccuracyTolerance:
    """Relative and absolute tolerances for a numerical comparison."""

    rtol: float
    atol: float

    def __post_init__(self) -> None:
        for name, value in (("rtol", self.rtol), ("atol", self.atol)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite, non-negative number, got {value!r}")


# The reference triton-ascend-kernels repository pins torch 2.7.1 and uses
# torch.testing.assert_close defaults as its dtype baseline, with explicit
# per-operator exceptions. Freeze those defaults here so a torch upgrade cannot
# silently change the vllm-ascend accuracy policy.
DEFAULT_DTYPE_TOLERANCES: Mapping[torch.dtype, AccuracyTolerance] = MappingProxyType(
    {
        torch.float16: AccuracyTolerance(rtol=1e-3, atol=1e-5),
        torch.bfloat16: AccuracyTolerance(rtol=1.6e-2, atol=1e-5),
        torch.float32: AccuracyTolerance(rtol=1.3e-6, atol=1e-5),
        torch.float64: AccuracyTolerance(rtol=1e-7, atol=1e-7),
    }
)
_EXACT_TOLERANCE = AccuracyTolerance(rtol=0.0, atol=0.0)
_EXACT_DTYPES = frozenset(
    dtype
    for name in ("bool", "uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64")
    if (dtype := getattr(torch, name, None)) is not None
)


def get_default_tolerance(dtype: torch.dtype) -> AccuracyTolerance:
    """Return the repository-wide baseline tolerance for ``dtype``.

    Floating-point dtypes outside the supported policy fail explicitly so a
    new dtype cannot silently inherit an unsuitable tolerance. Integer and
    boolean tensors are compared exactly.
    """

    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"dtype must be a torch.dtype, got {type(dtype).__name__}")
    if dtype in DEFAULT_DTYPE_TOLERANCES:
        return DEFAULT_DTYPE_TOLERANCES[dtype]
    if dtype in _EXACT_DTYPES:
        return _EXACT_TOLERANCE
    raise ValueError(
        f"No default accuracy tolerance is defined for {dtype}. "
        "Pass an explicit AccuracyTolerance override or extend DEFAULT_DTYPE_TOLERANCES."
    )


def assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    policy_dtype: torch.dtype | None = None,
    tolerance: AccuracyTolerance | None = None,
    exact: bool = False,
    equal_nan: bool = False,
    check_device: bool = True,
    check_dtype: bool = True,
    check_layout: bool = True,
    check_stride: bool = False,
    name: str | None = None,
    reason: str | None = None,
) -> None:
    """Compare an implementation result with its golden tensor under one policy.

    ``actual`` is the implementation under test and ``expected`` is the golden
    tensor. ``policy_dtype`` defaults to ``actual.dtype`` and should be supplied
    when tensors were promoted before comparison. Use ``exact=True`` for data
    movement, indices, masks, and other bit-exact contracts. Algorithm-specific
    exceptions must pass a named :class:`AccuracyTolerance` and explain why.
    """

    if not isinstance(actual, torch.Tensor) or not isinstance(expected, torch.Tensor):
        raise TypeError("actual and expected must both be torch.Tensor instances")
    if tolerance is not None and not isinstance(tolerance, AccuracyTolerance):
        raise TypeError(f"tolerance must be an AccuracyTolerance, got {type(tolerance).__name__}")
    if exact and tolerance is not None:
        raise ValueError("exact=True and a tolerance override are mutually exclusive")
    if tolerance is not None and not name:
        raise ValueError("a tolerance override requires a non-empty comparison name")
    if tolerance is not None and not reason:
        raise ValueError("a tolerance override requires a non-empty reason")

    policy_dtype = actual.dtype if policy_dtype is None else policy_dtype
    if not isinstance(policy_dtype, torch.dtype):
        raise TypeError(f"policy_dtype must be a torch.dtype, got {type(policy_dtype).__name__}")

    effective_tolerance = _EXACT_TOLERANCE if exact else tolerance or get_default_tolerance(policy_dtype)
    context = (
        f"{name or 'accuracy comparison'}: policy_dtype={policy_dtype}, "
        f"rtol={effective_tolerance.rtol}, atol={effective_tolerance.atol}, exact={exact}"
    )
    if reason:
        context = f"{context}; override_reason={reason}"

    def _with_context(default_message: str) -> str:
        return f"{context}\n{default_message}"

    torch.testing.assert_close(
        actual,
        expected,
        rtol=effective_tolerance.rtol,
        atol=effective_tolerance.atol,
        equal_nan=equal_nan,
        check_device=check_device,
        check_dtype=check_dtype,
        check_layout=check_layout,
        check_stride=check_stride,
        msg=_with_context,
    )


def assert_nrmse_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    max_nrmse: float,
    abs_guard: float = 1e-6,
    name: str,
    reason: str,
) -> None:
    """Compare recurrent outputs with an absolute guard and an NRMSE bound.

    This metric is intentionally separate from :func:`assert_close`:
    NRMSE constrains aggregate error and does not guarantee that every element
    satisfies an ``rtol``/``atol`` bound. Low-precision inputs are promoted to
    FP32 before reduction to avoid FP16/BF16 square overflow and underflow.
    Contract-compatible empty tensors pass, while any NaN or infinity fails.
    """

    if not isinstance(actual, torch.Tensor) or not isinstance(expected, torch.Tensor):
        raise TypeError("actual and expected must both be torch.Tensor instances")
    if not name:
        raise ValueError("an NRMSE comparison requires a non-empty name")
    if not reason:
        raise ValueError("an NRMSE comparison requires a non-empty reason")
    if actual.shape != expected.shape:
        raise AssertionError(f"{name}: shape mismatch: actual {actual.shape}, expected {expected.shape}")
    if actual.device != expected.device:
        raise AssertionError(f"{name}: device mismatch: actual {actual.device}, expected {expected.device}")
    if actual.dtype != expected.dtype:
        raise AssertionError(f"{name}: dtype mismatch: actual {actual.dtype}, expected {expected.dtype}")
    if not actual.is_floating_point() or not expected.is_floating_point():
        raise TypeError(f"{name}: NRMSE inputs must be floating-point tensors")
    for parameter_name, value in (("max_nrmse", max_nrmse), ("abs_guard", abs_guard)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{parameter_name} must be a finite, non-negative number, got {value!r}")
    if actual.numel() == 0:
        return
    if torch.isnan(actual).any():
        raise AssertionError(f"{name}: NaN detected in actual")
    if torch.isnan(expected).any():
        raise AssertionError(f"{name}: NaN detected in expected")
    if torch.isinf(actual).any():
        raise AssertionError(f"{name}: infinity detected in actual")
    if torch.isinf(expected).any():
        raise AssertionError(f"{name}: infinity detected in expected")

    reduction_dtype = torch.float64 if actual.dtype == torch.float64 else torch.float32
    actual_reduced = actual.detach().to(reduction_dtype)
    expected_reduced = expected.detach().to(reduction_dtype)
    difference = actual_reduced - expected_reduced
    max_abs_error = difference.flatten().abs().max().item()
    if max_abs_error <= abs_guard:
        return

    rmse = difference.flatten().square().mean().sqrt().item()
    expected_rms = expected_reduced.flatten().square().mean().sqrt().item()
    nrmse = rmse / (expected_rms + _NRMSE_DENOMINATOR_EPSILON)
    if not nrmse < max_nrmse:
        raise AssertionError(
            f"{name}: max_abs_error={max_abs_error:.6g}, nrmse={nrmse:.6g}, "
            f"max_nrmse={max_nrmse:.6g}; reason={reason}"
        )
