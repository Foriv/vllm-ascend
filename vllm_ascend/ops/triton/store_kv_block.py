# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Triton-Ascend implementation of the StoreKVBlock operator pair.

The module exposes compact serial and scan-based metadata implementations plus
a multi-AIV sparse representation.  The sparse path avoids cross-core global
compaction while remaining equivalent for the paired copy kernel.
"""

import torch
from vllm.triton_utils import tl, triton

_METADATA_BLOCK_SIZE = 256
_MULTI_AIV_METADATA_BLOCK_SIZE = 64
_PARALLEL_METADATA_MAX_SLOTS = 2048
_COPY_BLOCK_SIZE = 1024


@triton.jit(do_not_specialize=["num_slots", "output_capacity"])
def _store_kv_block_metadata_serial_kernel(
    slot_mapping_ptr,
    group_len_ptr,
    group_key_idx_ptr,
    group_key_cache_idx_ptr,
    num_slots,
    output_capacity,
    block_size: tl.constexpr,
    ZERO_BLOCK_SIZE: tl.constexpr,
):
    """Build compact contiguous-run metadata in slot-mapping order.

    A single program is deliberate here.  Both ``group_idx`` and the next
    input position depend on the preceding run, matching the AICPU reference
    implementation exactly without introducing a global prefix-sum workspace.
    """
    # Clearing the unused metadata and constructing the compact groups in the
    # same program avoids a separate device-kernel launch.  Metadata tensors
    # are small compared with the KV payload, so a vectorized loop here is
    # preferable to paying another launch for a parallel memset kernel.
    zero_offsets = tl.arange(0, ZERO_BLOCK_SIZE)
    for zero_start in range(0, output_capacity, ZERO_BLOCK_SIZE):
        offsets = zero_start + zero_offsets
        mask = offsets < output_capacity
        tl.store(group_len_ptr + offsets, 0, mask=mask)
        tl.store(group_key_idx_ptr + offsets, 0, mask=mask)
        tl.store(group_key_cache_idx_ptr + offsets, 0, mask=mask)

    slot_idx = 0
    group_idx = 0

    while slot_idx < num_slots:
        cache_slot = tl.load(slot_mapping_ptr + slot_idx)
        if cache_slot < 0:
            slot_idx += 1
        else:
            block_id = cache_slot // block_size
            group_end = slot_idx + 1
            keep_scanning = group_end < num_slots

            while (group_end < num_slots) & keep_scanning:
                previous_slot = tl.load(slot_mapping_ptr + group_end - 1)
                next_slot = tl.load(slot_mapping_ptr + group_end)
                is_consecutive = next_slot == previous_slot + 1
                is_same_block = next_slot // block_size == block_id
                keep_scanning = is_consecutive & is_same_block
                group_end += tl.where(keep_scanning, 1, 0)

            if group_idx < output_capacity:
                tl.store(group_len_ptr + group_idx, group_end - slot_idx)
                tl.store(group_key_idx_ptr + group_idx, slot_idx)
                tl.store(group_key_cache_idx_ptr + group_idx, cache_slot)

            group_idx += 1
            slot_idx = group_end


@triton.jit
def _store_kv_block_metadata_parallel_kernel(
    slot_mapping_ptr,
    group_len_ptr,
    group_key_idx_ptr,
    group_key_cache_idx_ptr,
    num_slots,
    output_capacity,
    block_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Detect group boundaries and compact them with an inclusive scan.

    Dynamic vector scatter through an ordinary ``tl.store`` is not supported
    reliably by the current Triton-Ascend backend.  ``atomic_max`` provides a
    supported scatter primitive for non-negative values; every destination
    still has exactly one writer, so this is not a contended reduction.
    """
    offsets = tl.arange(0, BLOCK_SIZE)
    slot_mask = offsets < num_slots
    output_mask = offsets < output_capacity

    tl.store(group_len_ptr + offsets, 0, mask=output_mask)
    tl.store(group_key_idx_ptr + offsets, 0, mask=output_mask)
    tl.store(group_key_cache_idx_ptr + offsets, 0, mask=output_mask)
    tl.debug_barrier()

    slots = tl.load(slot_mapping_ptr + offsets, mask=slot_mask, other=-1)
    valid = slot_mask & (slots >= 0)

    previous_mask = offsets > 0
    previous_offsets = tl.where(previous_mask, offsets - 1, 0)
    previous_slots = tl.load(slot_mapping_ptr + previous_offsets, mask=previous_mask, other=-1)
    continues_previous = (
        valid
        & (previous_slots >= 0)
        & (slots == previous_slots + 1)
        & (slots // block_size == previous_slots // block_size)
    )
    group_start = valid & ~continues_previous

    next_mask = offsets + 1 < num_slots
    next_slots = tl.load(slot_mapping_ptr + offsets + 1, mask=next_mask, other=-1)
    continues_next = (
        valid
        & (next_slots >= 0)
        & (next_slots == slots + 1)
        & (next_slots // block_size == slots // block_size)
    )
    group_end = valid & ~continues_next

    # Inclusive start scan gives every slot in a group the same compact id.
    group_ids = tl.cumsum(group_start.to(tl.int32), axis=0) - 1
    group_write_offsets = tl.where(group_ids >= 0, group_ids, 0)

    # Temporary group_len stores source end (exclusive).  Finalization turns
    # it into the actual length by subtracting group_key_idx (source start).
    tl.atomic_max(group_len_ptr + group_write_offsets, offsets + 1, mask=group_end)
    tl.atomic_max(group_key_idx_ptr + group_write_offsets, offsets, mask=group_start)
    tl.atomic_max(group_key_cache_idx_ptr + group_write_offsets, slots, mask=group_start)
    tl.debug_barrier()

    source_end = tl.load(group_len_ptr + offsets, mask=output_mask, other=0)
    group_start = tl.load(group_key_idx_ptr + offsets, mask=output_mask, other=0)
    cache_start = tl.load(group_key_cache_idx_ptr + offsets, mask=output_mask, other=0)
    valid_group = source_end > 0
    group_len = tl.where(valid_group, source_end - group_start, 0)
    group_start = tl.where(valid_group, group_start, 0)
    cache_start = tl.where(valid_group, cache_start, 0)
    tl.store(group_len_ptr + offsets, group_len, mask=output_mask)
    tl.store(group_key_idx_ptr + offsets, group_start, mask=output_mask)
    tl.store(group_key_cache_idx_ptr + offsets, cache_start, mask=output_mask)


@triton.jit(do_not_specialize=["num_slots", "output_capacity"])
def _store_kv_block_metadata_multi_aiv_kernel(
    slot_mapping_ptr,
    group_len_ptr,
    group_key_idx_ptr,
    group_key_cache_idx_ptr,
    num_slots,
    output_capacity,
    block_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Build sparse run metadata with one vector tile per program.

    Program instances are distributed across AIV cores.  Each program detects
    starts for a tile of source positions, then scans only while at least one
    lane owns an unfinished run.  Keeping each record at its source position
    removes the global compaction dependency between AIV cores.
    """
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    output_mask = offsets < output_capacity
    slot_mask = offsets < num_slots
    cache_slot = tl.load(slot_mapping_ptr + offsets, mask=slot_mask, other=-1)
    valid = slot_mask & (cache_slot >= 0)

    # Only the first token of a run performs the forward scan and emits a
    # record.  Reading the previous global position also handles runs that
    # cross AIV scheduling boundaries.
    has_previous = valid & (offsets > 0)
    previous_offsets = tl.where(has_previous, offsets - 1, 0)
    previous_slot = tl.load(slot_mapping_ptr + previous_offsets, mask=has_previous, other=-1)
    continues_previous = (
        has_previous
        & (previous_slot >= 0)
        & (cache_slot == previous_slot + 1)
        & (cache_slot // block_size == previous_slot // block_size)
    )
    group_start = valid & ~continues_previous

    block_id = cache_slot // block_size
    group_len = tl.where(group_start, 1, 0)
    scan_distance = 1
    active = group_start & (offsets + scan_distance < num_slots)
    has_active = tl.sum(active.to(tl.int32), axis=0) > 0
    while (scan_distance < block_size) & has_active:
        next_offsets = offsets + scan_distance
        next_mask = active & (next_offsets < num_slots)
        next_slot = tl.load(slot_mapping_ptr + next_offsets, mask=next_mask, other=-1)
        continues = (
            next_mask
            & (next_slot == cache_slot + scan_distance)
            & (next_slot // block_size == block_id)
        )
        group_len += continues.to(tl.int32)
        active = continues
        scan_distance += 1
        has_active = tl.sum(active.to(tl.int32), axis=0) > 0

    tl.store(group_len_ptr + offsets, group_len, mask=output_mask)
    tl.store(group_key_idx_ptr + offsets, tl.where(group_start, offsets, 0), mask=output_mask)
    tl.store(group_key_cache_idx_ptr + offsets, tl.where(group_start, cache_slot, 0), mask=output_mask)


@triton.jit(do_not_specialize=["token_size"])
def _store_kv_block_kernel(
    key_ptr,
    key_cache_ptr,
    group_len_ptr,
    group_key_idx_ptr,
    group_key_cache_idx_ptr,
    token_size,
    BLOCK_SIZE: tl.constexpr,
):
    group_idx = tl.program_id(0)
    group_len = tl.load(group_len_ptr + group_idx)
    source_token_idx = tl.load(group_key_idx_ptr + group_idx)
    cache_slot_idx = tl.load(group_key_cache_idx_ptr + group_idx)

    if (group_len <= 0) | (source_token_idx < 0) | (cache_slot_idx < 0):
        return

    copy_elements = group_len.to(tl.int64) * token_size
    source_start = source_token_idx.to(tl.int64) * token_size
    cache_start = cache_slot_idx.to(tl.int64) * token_size
    offsets = tl.arange(0, BLOCK_SIZE)

    for copied in range(0, copy_elements, BLOCK_SIZE):
        copy_offsets = copied + offsets
        mask = copy_offsets < copy_elements
        values = tl.load(key_ptr + source_start + copy_offsets, mask=mask)
        tl.store(key_cache_ptr + cache_start + copy_offsets, values, mask=mask)


def _validate_metadata_tensors(
    slot_mapping: torch.Tensor,
    group_len: torch.Tensor,
    group_key_idx: torch.Tensor,
    group_key_cache_idx: torch.Tensor,
    block_size: int,
) -> None:
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if slot_mapping.dtype != torch.int32:
        raise TypeError(f"slot_mapping must have dtype torch.int32, got {slot_mapping.dtype}")
    if slot_mapping.ndim != 1:
        raise ValueError(f"slot_mapping must be 1-D, got shape {tuple(slot_mapping.shape)}")
    if not slot_mapping.is_contiguous():
        raise ValueError("slot_mapping must be contiguous")

    outputs = (group_len, group_key_idx, group_key_cache_idx)
    output_capacity = group_len.numel()
    for output in outputs:
        if output.dtype != torch.int32:
            raise TypeError(f"metadata outputs must have dtype torch.int32, got {output.dtype}")
        if output.ndim != 1:
            raise ValueError(f"metadata outputs must be 1-D, got shape {tuple(output.shape)}")
        if not output.is_contiguous():
            raise ValueError("metadata outputs must be contiguous")
        if output.device != slot_mapping.device:
            raise ValueError("slot_mapping and metadata outputs must be on the same device")
        if output.numel() != output_capacity:
            raise ValueError("all metadata outputs must have the same capacity")

    # The number of groups is data-dependent.  Requiring worst-case capacity
    # avoids a device-to-host synchronization just to discover the exact count.
    if output_capacity < slot_mapping.numel():
        raise ValueError(
            "metadata output capacity must be at least slot_mapping.numel(); "
            f"got capacity={output_capacity}, num_slots={slot_mapping.numel()}"
        )


def store_kv_block_metadata_triton_serial(
    slot_mapping: torch.Tensor,
    group_len: torch.Tensor,
    group_key_idx: torch.Tensor,
    group_key_cache_idx: torch.Tensor,
    block_size: int,
) -> None:
    """Populate metadata with the original serial Triton state machine.

    The output tensors must have capacity for the worst case of one group per
    slot.  This keeps validation asynchronous and prevents silent truncation.
    """
    _validate_metadata_tensors(
        slot_mapping,
        group_len,
        group_key_idx,
        group_key_cache_idx,
        block_size,
    )

    output_capacity = group_len.numel()
    if output_capacity == 0:
        return

    _store_kv_block_metadata_serial_kernel[(1,)](
        slot_mapping,
        group_len,
        group_key_idx,
        group_key_cache_idx,
        slot_mapping.numel(),
        output_capacity,
        block_size,
        ZERO_BLOCK_SIZE=_METADATA_BLOCK_SIZE,
    )


def store_kv_block_metadata_triton_parallel(
    slot_mapping: torch.Tensor,
    group_len: torch.Tensor,
    group_key_idx: torch.Tensor,
    group_key_cache_idx: torch.Tensor,
    block_size: int,
) -> None:
    """Experimentally populate metadata with parallel detection and scan.

    A single vectorized program detects all boundaries and performs the scan.
    Inputs larger than the validated vector width use the serial baseline until
    a hierarchical multi-program scan is available.
    """
    _validate_metadata_tensors(
        slot_mapping,
        group_len,
        group_key_idx,
        group_key_cache_idx,
        block_size,
    )

    num_slots = slot_mapping.numel()
    output_capacity = group_len.numel()
    if output_capacity == 0:
        return
    if max(num_slots, output_capacity) > _PARALLEL_METADATA_MAX_SLOTS:
        store_kv_block_metadata_triton_serial(
            slot_mapping,
            group_len,
            group_key_idx,
            group_key_cache_idx,
            block_size,
        )
        return

    vector_size = triton.next_power_of_2(max(num_slots, output_capacity))
    _store_kv_block_metadata_parallel_kernel[(1,)](
        slot_mapping,
        group_len,
        group_key_idx,
        group_key_cache_idx,
        num_slots,
        output_capacity,
        block_size,
        BLOCK_SIZE=vector_size,
    )


def store_kv_block_metadata_triton_multi_aiv(
    slot_mapping: torch.Tensor,
    group_len: torch.Tensor,
    group_key_idx: torch.Tensor,
    group_key_cache_idx: torch.Tensor,
    block_size: int,
) -> None:
    """Populate sparse, copy-equivalent metadata across multiple AIV cores.

    Unlike the compact AICPU representation, records stay at their source
    token positions.  The paired StoreKVBlock copy kernel already treats
    ``group_len == 0`` as inactive.  Sixty-four source positions form one
    program tile, and the grid distributes those tiles across available AIV
    cores.  Avoiding compact output removes the cross-AIV prefix sum.
    """
    _validate_metadata_tensors(
        slot_mapping,
        group_len,
        group_key_idx,
        group_key_cache_idx,
        block_size,
    )

    num_slots = slot_mapping.numel()
    output_capacity = group_len.numel()
    if output_capacity == 0:
        return

    grid = (triton.cdiv(output_capacity, _MULTI_AIV_METADATA_BLOCK_SIZE),)
    _store_kv_block_metadata_multi_aiv_kernel[grid](
        slot_mapping,
        group_len,
        group_key_idx,
        group_key_cache_idx,
        num_slots,
        output_capacity,
        block_size,
        BLOCK_SIZE=_MULTI_AIV_METADATA_BLOCK_SIZE,
    )


def store_kv_block_metadata_triton(
    slot_mapping: torch.Tensor,
    group_len: torch.Tensor,
    group_key_idx: torch.Tensor,
    group_key_cache_idx: torch.Tensor,
    block_size: int,
) -> None:
    """Populate StoreKVBlock metadata using the fastest validated path."""
    store_kv_block_metadata_triton_serial(
        slot_mapping,
        group_len,
        group_key_idx,
        group_key_cache_idx,
        block_size,
    )


def _validate_store_tensors(
    key: torch.Tensor,
    key_cache: torch.Tensor,
    group_len: torch.Tensor,
    group_key_idx: torch.Tensor,
    group_key_cache_idx: torch.Tensor,
    block_size: int,
) -> int:
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if key.ndim == 0 or key.shape[0] == 0:
        raise ValueError("key must contain at least one token")
    if key.dtype != key_cache.dtype:
        raise TypeError(f"key and key_cache must have the same dtype, got {key.dtype} and {key_cache.dtype}")
    if key.device != key_cache.device:
        raise ValueError("key and key_cache must be on the same device")
    if not key.is_contiguous() or not key_cache.is_contiguous():
        raise ValueError("key and key_cache must be contiguous")

    metadata = (group_len, group_key_idx, group_key_cache_idx)
    num_groups = group_len.numel()
    for tensor in metadata:
        if tensor.dtype != torch.int32:
            raise TypeError(f"metadata tensors must have dtype torch.int32, got {tensor.dtype}")
        if tensor.ndim != 1 or tensor.numel() != num_groups:
            raise ValueError("metadata tensors must be 1-D and have the same number of elements")
        if tensor.device != key.device:
            raise ValueError("key, key_cache, and metadata tensors must be on the same device")
        if not tensor.is_contiguous():
            raise ValueError("metadata tensors must be contiguous")

    token_size = key.numel() // key.shape[0]
    if key_cache.numel() % token_size != 0:
        raise ValueError(
            "key_cache size must be an integer multiple of the flattened token size; "
            f"got cache elements={key_cache.numel()}, token_size={token_size}"
        )
    return token_size


def store_kv_block_triton(
    key: torch.Tensor,
    key_cache: torch.Tensor,
    group_len: torch.Tensor,
    group_key_idx: torch.Tensor,
    group_key_cache_idx: torch.Tensor,
    block_size: int,
) -> None:
    """Copy grouped token ranges from ``key`` into ``key_cache`` in place."""
    token_size = _validate_store_tensors(
        key,
        key_cache,
        group_len,
        group_key_idx,
        group_key_cache_idx,
        block_size,
    )
    if group_len.numel() == 0:
        return

    _store_kv_block_kernel[(group_len.numel(),)](
        key,
        key_cache,
        group_len,
        group_key_idx,
        group_key_cache_idx,
        token_size,
        BLOCK_SIZE=_COPY_BLOCK_SIZE,
    )


__all__ = [
    "store_kv_block_metadata_triton",
    "store_kv_block_metadata_triton_multi_aiv",
    "store_kv_block_metadata_triton_parallel",
    "store_kv_block_metadata_triton_serial",
    "store_kv_block_triton",
]
