# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.ops.triton.store_kv_block import (
    store_kv_block_metadata_triton,
    store_kv_block_metadata_triton_multi_aiv,
    store_kv_block_metadata_triton_parallel,
    store_kv_block_metadata_triton_serial,
    store_kv_block_triton,
)

METADATA_IMPLEMENTATIONS = (
    store_kv_block_metadata_triton_serial,
    store_kv_block_metadata_triton_parallel,
    store_kv_block_metadata_triton,
)


def _metadata_reference(slot_mapping: list[int], block_size: int) -> tuple[list[int], list[int], list[int]]:
    group_len = [0] * len(slot_mapping)
    group_key_idx = [0] * len(slot_mapping)
    group_key_cache_idx = [0] * len(slot_mapping)
    slot_idx = 0
    group_idx = 0

    while slot_idx < len(slot_mapping):
        cache_slot = slot_mapping[slot_idx]
        if cache_slot < 0:
            slot_idx += 1
            continue

        block_id = cache_slot // block_size
        group_end = slot_idx + 1
        while (
            group_end < len(slot_mapping)
            and slot_mapping[group_end] // block_size == block_id
            and slot_mapping[group_end] == slot_mapping[group_end - 1] + 1
        ):
            group_end += 1

        group_len[group_idx] = group_end - slot_idx
        group_key_idx[group_idx] = slot_idx
        group_key_cache_idx[group_idx] = cache_slot
        group_idx += 1
        slot_idx = group_end

    return group_len, group_key_idx, group_key_cache_idx


def _sparse_metadata_reference(slot_mapping: list[int], block_size: int) -> tuple[list[int], list[int], list[int]]:
    compact_group_len, compact_key_idx, compact_cache_idx = _metadata_reference(slot_mapping, block_size)
    group_len = [0] * len(slot_mapping)
    group_key_idx = [0] * len(slot_mapping)
    group_key_cache_idx = [0] * len(slot_mapping)

    for length, key_idx, cache_idx in zip(compact_group_len, compact_key_idx, compact_cache_idx):
        if length == 0:
            break
        group_len[key_idx] = length
        group_key_idx[key_idx] = key_idx
        group_key_cache_idx[key_idx] = cache_idx

    return group_len, group_key_idx, group_key_cache_idx


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
@pytest.mark.parametrize("metadata_impl", METADATA_IMPLEMENTATIONS)
def test_store_kv_block_metadata_triton(slot_mapping: list[int], block_size: int, metadata_impl):
    device = "npu:0"
    slots = torch.tensor(slot_mapping, dtype=torch.int32, device=device)
    group_len = torch.empty_like(slots)
    group_key_idx = torch.empty_like(slots)
    group_key_cache_idx = torch.empty_like(slots)

    metadata_impl(
        slots,
        group_len,
        group_key_idx,
        group_key_cache_idx,
        block_size,
    )

    expected = _metadata_reference(slot_mapping, block_size)
    torch.testing.assert_close(group_len.cpu(), torch.tensor(expected[0], dtype=torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(group_key_idx.cpu(), torch.tensor(expected[1], dtype=torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(group_key_cache_idx.cpu(), torch.tensor(expected[2], dtype=torch.int32), rtol=0, atol=0)


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
def test_store_kv_block_metadata_triton_multi_aiv(slot_mapping: list[int], block_size: int):
    slots = torch.tensor(slot_mapping, dtype=torch.int32, device="npu:0")
    group_len = torch.empty_like(slots)
    group_key_idx = torch.empty_like(slots)
    group_key_cache_idx = torch.empty_like(slots)

    store_kv_block_metadata_triton_multi_aiv(
        slots,
        group_len,
        group_key_idx,
        group_key_cache_idx,
        block_size,
    )

    expected = _sparse_metadata_reference(slot_mapping, block_size)
    torch.testing.assert_close(group_len.cpu(), torch.tensor(expected[0], dtype=torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(group_key_idx.cpu(), torch.tensor(expected[1], dtype=torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(group_key_cache_idx.cpu(), torch.tensor(expected[2], dtype=torch.int32), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.int8, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "metadata_impl",
    [store_kv_block_metadata_triton, store_kv_block_metadata_triton_multi_aiv],
)
def test_store_kv_block_triton(dtype: torch.dtype, metadata_impl):
    device = "npu:0"
    block_size = 4
    slot_mapping = [-1, 2, 3, 4, 9, 10]
    num_tokens = len(slot_mapping)
    token_shape = (2, 8)
    cache_shape = (4, block_size, *token_shape)

    if dtype == torch.int8:
        key = torch.randint(-8, 8, (num_tokens, *token_shape), dtype=dtype, device=device)
        key_cache = torch.randint(-8, 8, cache_shape, dtype=dtype, device=device)
    else:
        key = torch.randn((num_tokens, *token_shape), dtype=dtype, device=device)
        key_cache = torch.randn(cache_shape, dtype=dtype, device=device)

    expected = key_cache.clone().view(-1, *token_shape)
    for token_idx, cache_slot in enumerate(slot_mapping):
        if cache_slot >= 0:
            expected[cache_slot].copy_(key[token_idx])

    slots = torch.tensor(slot_mapping, dtype=torch.int32, device=device)
    group_len = torch.empty_like(slots)
    group_key_idx = torch.empty_like(slots)
    group_key_cache_idx = torch.empty_like(slots)
    metadata_impl(
        slots,
        group_len,
        group_key_idx,
        group_key_cache_idx,
        block_size,
    )
    store_kv_block_triton(
        key,
        key_cache,
        group_len,
        group_key_idx,
        group_key_cache_idx,
        block_size,
    )

    torch.testing.assert_close(key_cache, expected.view(cache_shape), rtol=0, atol=0)
