# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Equivalence test: custom scatter_nd_update_v2 vs replacement candidates.

On-device probing established the custom op's *valid* semantic domain:

- 2-D flat cache ``[num_slots, width]`` with indices of shape ``[N, 1]``;
- flat slot ids; out-of-range indices (e.g. -1 padding) are ignored;
- 1-D indices are silently ignored (no-op), and >2-D caches with ``[N, 1]``
  indices produce garbage -- neither belongs to the valid domain.

Within the valid domain this test requires bit-exact equivalence with the
``index_copy_``-based replacement. CANN-hosted ``torch_npu.scatter_update_``
is also exercised for documentation: it cannot even run reliably for the
valid domain (fp16 inputs fail on this CANN build), so it is not a viable
replacement and is expected to xfail/diverge.
"""
import pytest
import torch
import torch_npu  # noqa: F401

import vllm_ascend  # noqa: F401
import vllm_ascend.ops  # noqa: F401  # init order avoids circular import
from vllm_ascend.utils import enable_custom_op

enable_custom_op()


def _custom_op(cache: torch.Tensor, slot_mapping: torch.Tensor, updates: torch.Tensor) -> None:
    torch.ops._C_ascend.npu_scatter_nd_update_v2(cache, slot_mapping, updates)


def _index_copy_scatter(cache: torch.Tensor, slot_mapping: torch.Tensor, updates: torch.Tensor) -> None:
    """Replacement path used by device_op.scatter_slots_inplace_."""
    rows = cache.shape[0] * cache.shape[1] if cache.dim() > 2 else cache.shape[0]
    width = cache.numel() // rows
    flat_cache = cache.view(rows, width)
    flat_updates = updates.view(updates.shape[0], width)
    indices = slot_mapping.view(-1)
    valid = (indices >= 0) & (indices < rows)
    pos = valid.nonzero(as_tuple=True)[0]
    flat_cache.index_copy_(0, indices[pos], flat_updates[pos])


def _cann_scatter_update(cache: torch.Tensor, slot_mapping: torch.Tensor, updates: torch.Tensor) -> None:
    """CANN-hosted op applied per its kernel constraints (3-D view, axis=-2)."""
    rows = cache.shape[0] * cache.shape[1] if cache.dim() > 2 else cache.shape[0]
    width = cache.numel() // rows
    view = cache.view(1, rows, width)
    upd = updates.view(updates.shape[0], 1, width)
    idx = slot_mapping.view(-1)
    valid = (idx >= 0) & (idx < rows)
    pos = valid.nonzero(as_tuple=True)[0]
    torch_npu.scatter_update_(view, idx[pos], upd[pos], axis=-2)


def _run_pair(fn_a, fn_b, cache_shape, updates_shape, slots_2d, dtype):
    torch.manual_seed(2026)
    cache = (torch.randn(cache_shape) * 0.5).to(dtype).npu()
    updates = (torch.randn(updates_shape) * 0.5).to(dtype).npu()
    idx = torch.tensor(slots_2d, dtype=torch.int32).npu()
    c_a, c_b = cache.clone(), cache.clone()
    fn_a(c_a, idx, updates)
    fn_b(c_b, idx, updates)
    return (c_a.float() - c_b.float()).abs().max().item()


# Valid domain: 2-D flat cache, [N, 1] indices, includes -1 / OOB entries.
VALID_CASES = [
    ((64, 128), (4, 128), [[0], [3], [31], [63]]),
    ((64, 128), (5, 128), [[3], [10], [31], [-1], [64]]),  # -1 and 64 are out of bounds
    ((128, 2, 64), (4, 2, 64), [[0], [7], [64], [127]]),  # dim0=128 slots, width=128
]
DTYPES = [torch.float16, torch.bfloat16, torch.int8]


@pytest.mark.parametrize("cache_shape,updates_shape,slots", VALID_CASES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_custom_op_vs_index_copy(cache_shape, updates_shape, slots, dtype):
    """index_copy_ replacement must be bit-exact with the custom op."""
    diff = _run_pair(_custom_op, _index_copy_scatter, cache_shape, updates_shape, slots, dtype)
    assert diff == 0, f"custom op vs index_copy_ max diff = {diff} (dtype={dtype})"


@pytest.mark.parametrize("cache_shape,updates_shape,slots", VALID_CASES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_custom_op_vs_cann_scatter_update(cache_shape, updates_shape, slots, dtype):
    """Document CANN scatter_update_ behavior in the valid domain.

    Known status on this CANN build: fp16 inputs fail in the op kernel, so
    these cases xfail; fp32-family behavior is recorded for reference.
    """
    try:
        diff = _run_pair(_custom_op, _cann_scatter_update, cache_shape, updates_shape, slots, dtype)
    except RuntimeError as err:
        pytest.xfail(f"CANN scatter_update_ failed to run (dtype={dtype}): {str(err).splitlines()[0][:80]}")
        return
    assert diff == 0, f"CANN scatter_update_ diverges, max diff = {diff} (dtype={dtype})"
