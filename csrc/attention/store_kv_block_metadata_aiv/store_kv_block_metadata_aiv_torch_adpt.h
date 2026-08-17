/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#ifndef STORE_KV_BLOCK_METADATA_AIV_TORCH_ADPT_H
#define STORE_KV_BLOCK_METADATA_AIV_TORCH_ADPT_H

namespace vllm_ascend {
namespace {
void check_store_kv_block_metadata_aiv_args(
    const at::Tensor& slot_mapping,
    const at::Tensor& group_len,
    const at::Tensor& group_key_idx,
    const at::Tensor& group_key_cache_idx,
    int64_t block_size)
{
    TORCH_CHECK(slot_mapping.numel() > 0, "Tensor slot_mapping is empty.");
    TORCH_CHECK(slot_mapping.scalar_type() == at::ScalarType::Int,
                "slot_mapping must have dtype int32.");
    TORCH_CHECK(group_len.scalar_type() == at::ScalarType::Int &&
                group_key_idx.scalar_type() == at::ScalarType::Int &&
                group_key_cache_idx.scalar_type() == at::ScalarType::Int,
                "metadata outputs must have dtype int32.");
    TORCH_CHECK(group_len.numel() >= slot_mapping.numel() &&
                group_key_idx.numel() == group_len.numel() &&
                group_key_cache_idx.numel() == group_len.numel(),
                "metadata outputs must have equal capacity and capacity must be >= slot_mapping length.");
    TORCH_CHECK(block_size > 0, "block_size must be positive, but got ", block_size);
}
}  // namespace

void store_kv_block_metadata_aiv_serial(
    const at::Tensor& slot_mapping,
    const at::Tensor& group_len,
    const at::Tensor& group_key_idx,
    const at::Tensor& group_key_cache_idx,
    int64_t block_size)
{
    check_store_kv_block_metadata_aiv_args(
        slot_mapping, group_len, group_key_idx, group_key_cache_idx, block_size);
    int64_t mode = 0;
    EXEC_NPU_CMD(aclnnStoreKvBlockMetadataAiv, slot_mapping, group_len, group_key_idx,
                 group_key_cache_idx, block_size, mode);
}

void store_kv_block_metadata_aiv_multi(
    const at::Tensor& slot_mapping,
    const at::Tensor& group_len,
    const at::Tensor& group_key_idx,
    const at::Tensor& group_key_cache_idx,
    int64_t block_size)
{
    check_store_kv_block_metadata_aiv_args(
        slot_mapping, group_len, group_key_idx, group_key_cache_idx, block_size);
    int64_t mode = 1;
    EXEC_NPU_CMD(aclnnStoreKvBlockMetadataAiv, slot_mapping, group_len, group_key_idx,
                 group_key_cache_idx, block_size, mode);
}
}  // namespace vllm_ascend

#endif  // STORE_KV_BLOCK_METADATA_AIV_TORCH_ADPT_H
