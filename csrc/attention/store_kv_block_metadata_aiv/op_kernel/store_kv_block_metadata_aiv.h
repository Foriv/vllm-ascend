/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#ifndef STORE_KV_BLOCK_METADATA_AIV_H
#define STORE_KV_BLOCK_METADATA_AIV_H

#include "kernel_operator.h"

namespace StoreKvBlockMetadataAiv {
using namespace AscendC;

struct StoreKvBlockMetadataAivTilingData {
    uint32_t numSlots;
    uint32_t outputCapacity;
    uint32_t blockSize;
    uint32_t mode;
    uint32_t usedCoreNum;
};

class StoreKvBlockMetadataAivKernel {
 public:
    __aicore__ inline void Init(GM_ADDR slotMapping, GM_ADDR groupLen, GM_ADDR groupKeyIdx,
                                GM_ADDR groupKeyCacheIdx,
                                const StoreKvBlockMetadataAivTilingData* tilingData)
    {
        slotMappingGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(slotMapping));
        groupLenGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(groupLen));
        groupKeyIdxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(groupKeyIdx));
        groupKeyCacheIdxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(groupKeyCacheIdx));
        numSlots_ = tilingData->numSlots;
        outputCapacity_ = tilingData->outputCapacity;
        blockSize_ = tilingData->blockSize;
        usedCoreNum_ = tilingData->usedCoreNum;
    }

    __aicore__ inline void ProcessSerial()
    {
        // On A2's separated AIC/AIV architecture, one physical AI Core owns two
        // AIV sub-cores.  BlockDim=1 therefore starts both sub-cores; keep only
        // sub-core 0 so this path is a genuine single-AIV serial baseline.
        if (GetSubBlockIdx() != 0) {
            return;
        }
        ZeroRange(0, outputCapacity_);

        uint32_t slotIndex = 0;
        uint32_t groupIndex = 0;
        while (slotIndex < numSlots_) {
            int32_t cacheSlot = slotMappingGm_.GetValue(slotIndex);
            if (cacheSlot < 0) {
                ++slotIndex;
                continue;
            }

            uint32_t groupEnd = FindGroupEnd(slotIndex, cacheSlot);
            groupLenGm_.SetValue(groupIndex, static_cast<int32_t>(groupEnd - slotIndex));
            groupKeyIdxGm_.SetValue(groupIndex, static_cast<int32_t>(slotIndex));
            groupKeyCacheIdxGm_.SetValue(groupIndex, cacheSlot);
            slotIndex = groupEnd;
            ++groupIndex;
        }
    }

    __aicore__ inline void ProcessMultiCore()
    {
        // On A2, GetBlockIdx() is the AI-Core-combination index encoded at a
        // stride of two for AIV execution, while GetSubBlockIdx() selects one
        // of the two vector sub-cores.  Their sum is the dense logical AIV id:
        //   (0, sub 0/1), (2, sub 0/1), ... -> 0, 1, 2, 3, ...
        uint32_t coreIndex = GetBlockIdx() + GetSubBlockIdx();
        if (coreIndex >= usedCoreNum_) {
            return;
        }

        constexpr uint32_t INT32_PER_CACHE_LINE = 16;
        uint32_t outputLineCount =
            (outputCapacity_ + INT32_PER_CACHE_LINE - 1) / INT32_PER_CACHE_LINE;
        for (uint32_t lineIndex = coreIndex; lineIndex < outputLineCount;
             lineIndex += usedCoreNum_) {
            uint32_t lineStart = lineIndex * INT32_PER_CACHE_LINE;
            uint32_t lineEnd = lineStart + INT32_PER_CACHE_LINE;
            if (lineEnd > outputCapacity_) {
                lineEnd = outputCapacity_;
            }
            for (uint32_t slotIndex = lineStart; slotIndex < lineEnd; ++slotIndex) {
                int32_t groupLength = 0;
                int32_t groupKeyIndex = 0;
                int32_t groupCacheIndex = 0;
                if (slotIndex < numSlots_) {
                    int32_t cacheSlot = slotMappingGm_.GetValue(slotIndex);
                    if (cacheSlot >= 0 && IsGroupStart(slotIndex, cacheSlot)) {
                        uint32_t groupEnd = FindGroupEnd(slotIndex, cacheSlot);
                        groupLength = static_cast<int32_t>(groupEnd - slotIndex);
                        groupKeyIndex = static_cast<int32_t>(slotIndex);
                        groupCacheIndex = cacheSlot;
                    }
                }

                // Sparse layout: write a group record at its source-token index.
                // A complete cache line belongs to this core, which avoids
                // cross-AIV scalar-cache writeback conflicts.
                groupLenGm_.SetValue(slotIndex, groupLength);
                groupKeyIdxGm_.SetValue(slotIndex, groupKeyIndex);
                groupKeyCacheIdxGm_.SetValue(slotIndex, groupCacheIndex);
            }
        }
    }

 private:
    __aicore__ inline void ZeroRange(uint32_t start, uint32_t end)
    {
        for (uint32_t index = start; index < end; ++index) {
            groupLenGm_.SetValue(index, 0);
            groupKeyIdxGm_.SetValue(index, 0);
            groupKeyCacheIdxGm_.SetValue(index, 0);
        }
    }

    __aicore__ inline bool IsGroupStart(uint32_t slotIndex, int32_t cacheSlot)
    {
        if (slotIndex == 0) {
            return true;
        }
        int32_t previousSlot = slotMappingGm_.GetValue(slotIndex - 1);
        if (previousSlot < 0) {
            return true;
        }
        return cacheSlot / static_cast<int32_t>(blockSize_) !=
                   previousSlot / static_cast<int32_t>(blockSize_) ||
               static_cast<int64_t>(cacheSlot) != static_cast<int64_t>(previousSlot) + 1;
    }

    __aicore__ inline uint32_t FindGroupEnd(uint32_t slotIndex, int32_t cacheSlot)
    {
        int32_t blockId = cacheSlot / static_cast<int32_t>(blockSize_);
        int32_t previousSlot = cacheSlot;
        uint32_t groupEnd = slotIndex + 1;
        while (groupEnd < numSlots_) {
            int32_t nextSlot = slotMappingGm_.GetValue(groupEnd);
            if (nextSlot < 0 || nextSlot / static_cast<int32_t>(blockSize_) != blockId ||
                static_cast<int64_t>(nextSlot) != static_cast<int64_t>(previousSlot) + 1) {
                break;
            }
            previousSlot = nextSlot;
            ++groupEnd;
        }
        return groupEnd;
    }

    uint32_t numSlots_ = 0;
    uint32_t outputCapacity_ = 0;
    uint32_t blockSize_ = 0;
    uint32_t usedCoreNum_ = 1;
    GlobalTensor<int32_t> slotMappingGm_;
    GlobalTensor<int32_t> groupLenGm_;
    GlobalTensor<int32_t> groupKeyIdxGm_;
    GlobalTensor<int32_t> groupKeyCacheIdxGm_;
};
}  // namespace StoreKvBlockMetadataAiv

#endif  // STORE_KV_BLOCK_METADATA_AIV_H
