# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401


def _new_metadata(slot_mapping: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.empty_like(slot_mapping),
        torch.empty_like(slot_mapping),
        torch.empty_like(slot_mapping),
    )


def _run_metadata(
    op_name: str,
    slot_mapping: torch.Tensor,
    metadata: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    block_size: int,
) -> None:
    getattr(torch.ops._C_ascend, op_name)(slot_mapping, *metadata, block_size)


def _active_groups(metadata: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> list[tuple[int, int, int]]:
    group_len, group_key_idx, group_key_cache_idx = (tensor.cpu().tolist() for tensor in metadata)
    return sorted(
        (key_idx, cache_idx, length)
        for length, key_idx, cache_idx in zip(group_len, group_key_idx, group_key_cache_idx)
        if length > 0
    )


@pytest.mark.parametrize(
    ("slot_mapping", "block_size"),
    [
        ([2, 3, 4, 5, 6, 7], 4),
        ([-1, 0, 1, -1, 6, 7, 8, 10], 4),
        ([7, 8, 9, 3, 4, -1, 12], 8),
        ([-1, -1, -1], 16),
        (list(range(42, 58)), 128),
        ([2 * token_idx for token_idx in range(128)], 128),
    ],
)
def test_store_kv_block_metadata_aiv(slot_mapping: list[int], block_size: int) -> None:
    slots = torch.tensor(slot_mapping, dtype=torch.int32, device="npu:0")
    aicpu = _new_metadata(slots)
    aiv_serial = _new_metadata(slots)
    aiv_multi = _new_metadata(slots)

    _run_metadata("store_kv_block_metadata", slots, aicpu, block_size)
    _run_metadata("store_kv_block_metadata_aiv_serial", slots, aiv_serial, block_size)
    _run_metadata("store_kv_block_metadata_aiv_multi", slots, aiv_multi, block_size)
    torch.npu.synchronize()

    # Single-AIV mode preserves the compact AICPU layout exactly.
    for expected, actual in zip(aicpu, aiv_serial):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)

    # Multi-AIV mode stores each record at its source-token index.  Its raw
    # layout is sparse, but the set of effective groups must be identical.
    assert _active_groups(aicpu) == _active_groups(aiv_multi)


@pytest.mark.parametrize("metadata_op", [
    "store_kv_block_metadata",
    "store_kv_block_metadata_aiv_serial",
    "store_kv_block_metadata_aiv_multi",
])
def test_store_kv_block_metadata_aiv_copy_equivalence(metadata_op: str) -> None:
    block_size = 4
    slot_mapping = [-1, 2, 3, 4, 9, 10]
    slots = torch.tensor(slot_mapping, dtype=torch.int32, device="npu:0")
    metadata = _new_metadata(slots)
    key = torch.arange(len(slot_mapping) * 16, dtype=torch.int32, device="npu:0").to(torch.int8).view(-1, 2, 8)
    key_cache = torch.zeros((4, block_size, 2, 8), dtype=torch.int8, device="npu:0")
    expected = key_cache.clone().view(-1, 2, 8)
    for token_index, cache_slot in enumerate(slot_mapping):
        if cache_slot >= 0:
            expected[cache_slot].copy_(key[token_index])

    _run_metadata(metadata_op, slots, metadata, block_size)
    torch.ops._C_ascend.store_kv_block(key, key_cache, *metadata, block_size)

    torch.testing.assert_close(key_cache.view_as(expected), expected, rtol=0, atol=0)

