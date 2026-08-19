# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Equivalence test: custom scatter_nd_update_v2 vs CANN-hosted scatter paths.

The custom op ``npu_scatter_nd_update_v2`` is planned for sunset because CANN
hosts scatter update natively (``torch_npu.scatter_update_``). This test
checks whether the two are actually equivalent for the paged-KV-cache scatter
semantics the custom op provides:

- cache viewed as [num_blocks * block_size, -1], updates written at flat slot
  positions from ``slot_mapping`` (1-D or [N, 1] indices);
- out-of-range indices (e.g. -1 padding slots) are ignored;
- fp16 / bf16 / int8 dtypes on 2D / 3D / 4D cache layouts.
"""
import pytest
import torch
import torch_npu  # noqa: F401

import vllm_ascend  # noqa: F401
import vllm_ascend.ops  # noqa: F401  # init order avoids circular import


def _custom_op(cache: torch.Tensor, slot_mapping: torch.Tensor, updates: torch.Tensor) -> None:
    torch.ops._C_ascend.npu_scatter_nd_update_v2(cache, slot_mapping, updates)


def _index_copy_scatter(cache: torch.Tensor, slot_mapping: torch.Tensor, updates: torch.Tensor) -> None:
    """Candidate replacement: flatten + index_copy_ + drop out-of-range indices."""
    rows = cache.shape[0] * cache.shape[1] if cache.dim() > 2 else cache.shape[0]
    width = cache.numel() // rows
    flat_cache = cache.view(rows, width)
    flat_updates = updates.view(updates.shape[0], width)
    indices = slot_mapping.view(-1)
    valid = (indices >= 0) & (indices < rows)
    pos = valid.nonzero(as_tuple=True)[0]
    flat_cache.index_copy_(0, indices[pos], flat_updates[pos])


def _cann_scatter_update(cache: torch.Tensor, slot_mapping: torch.Tensor, updates: torch.Tensor) -> None:
    """CANN-hosted op, applied the way its constraints allow for our layout."""
    rows = cache.shape[0] * cache.shape[1] if cache.dim() > 2 else cache.shape[0]
    width = cache.numel() // rows
    view = cache.view(1, rows, width)  # axis cannot be the first dim
    upd = updates.view(updates.shape[0], 1, width)
    idx = slot_mapping.view(-1)
    valid = (idx >= 0) & (idx < rows)
    pos = valid.nonzero(as_tuple=True)[0]
    torch_npu.scatter_update_(view, idx[pos], upd[pos], axis=-2)


def _run_pair(fn_a, fn_b, cache_shape, updates_shape, slots, dtype):
    torch.manual_seed(2026)
    cache = (torch.randn(cache_shape) * 0.5).to(dtype).npu()
    updates = (torch.randn(updates_shape) * 0.5).to(dtype).npu()
    idx = torch.tensor(slots, dtype=torch.int32).npu()
    c_a, c_b = cache.clone(), cache.clone()
    fn_a(c_a, idx, updates)
    fn_b(c_b, idx, updates)
    return (c_a.float() - c_b.float()).abs().max().item()


CASES = [
    ("4D", (4, 16, 2, 64), (6, 2, 64), [0, 5, 63, -1, 17, 100]),
    ("3D", (4, 16, 128), (5, 128), [0, 63, -1, 7, 64]),
    ("2D", (64, 128), (4, 128), [3, -1, 63, 64]),
]
DTYPES = [torch.float16, torch.bfloat16, torch.int8]


@pytest.mark.parametrize("layout,cache_shape,updates_shape,slots", CASES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_custom_op_vs_index_copy(layout, cache_shape, updates_shape, slots, dtype):
    """index_copy_ path must be bit-exact with the custom op."""
    diff = _run_pair(_custom_op, _index_copy_scatter, cache_shape, updates_shape, slots, dtype)
    assert diff == 0, f"{layout}/{dtype}: custom op vs index_copy_ max diff = {diff}"


@pytest.mark.parametrize("layout,cache_shape,updates_shape,slots", CASES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_custom_op_vs_cann_scatter_update(layout, cache_shape, updates_shape, slots, dtype):
    """Document whether CANN-hosted scatter_update_ matches the custom op."""
    try:
        diff = _run_pair(_custom_op, _cann_scatter_update, cache_shape, updates_shape, slots, dtype)
    except RuntimeError as err:
        pytest.xfail(f"{layout}/{dtype}: CANN scatter_update_ failed to run: {str(err).splitlines()[0][:80]}")
        return
    assert diff == 0, (
        f"{layout}/{dtype}: CANN scatter_update_ diverges from custom op, max diff = {diff}"
    )
