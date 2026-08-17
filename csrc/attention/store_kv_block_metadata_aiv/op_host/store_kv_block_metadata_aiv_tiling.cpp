/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <algorithm>
#include <cstdint>
#include "store_kv_block_metadata_aiv_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/error_log.h"

namespace optiling {
namespace {
constexpr uint32_t SLOT_MAPPING_INDEX = 0;
constexpr uint32_t GROUP_LEN_INDEX = 1;
constexpr uint32_t GROUP_KEY_IDX_INDEX = 2;
constexpr uint32_t GROUP_KEY_CACHE_IDX_INDEX = 3;
constexpr uint32_t BLOCK_SIZE_ATTR_INDEX = 0;
constexpr uint32_t MODE_ATTR_INDEX = 1;
constexpr int64_t SERIAL_MODE = 0;
constexpr int64_t MULTI_CORE_MODE = 1;
constexpr uint32_t INT32_PER_CACHE_LINE = 16;

ge::graphStatus GetVectorLength(gert::TilingContext* context, uint32_t index, uint32_t& length)
{
    auto shape = context->GetInputShape(index);
    OP_CHECK_NULL_WITH_CONTEXT(context, shape);
    const auto& storageShape = shape->GetStorageShape();
    if (storageShape.GetDimNum() != 1 || storageShape.GetDim(0) < 0) {
        OP_LOGE(context->GetNodeName(), "input %u must be a one-dimensional tensor", index);
        return ge::GRAPH_FAILED;
    }
    length = static_cast<uint32_t>(storageShape.GetDim(0));
    return ge::GRAPH_SUCCESS;
}
}  // namespace

static ge::graphStatus StoreKvBlockMetadataAivTilingFunc(gert::TilingContext* context)
{
    uint32_t numSlots = 0;
    uint32_t outputCapacity = 0;
    uint32_t keyIdxCapacity = 0;
    uint32_t cacheIdxCapacity = 0;
    if (GetVectorLength(context, SLOT_MAPPING_INDEX, numSlots) != ge::GRAPH_SUCCESS ||
        GetVectorLength(context, GROUP_LEN_INDEX, outputCapacity) != ge::GRAPH_SUCCESS ||
        GetVectorLength(context, GROUP_KEY_IDX_INDEX, keyIdxCapacity) != ge::GRAPH_SUCCESS ||
        GetVectorLength(context, GROUP_KEY_CACHE_IDX_INDEX, cacheIdxCapacity) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (outputCapacity < numSlots || keyIdxCapacity != outputCapacity || cacheIdxCapacity != outputCapacity) {
        OP_LOGE(context->GetNodeName(),
                "metadata outputs must have equal capacity and capacity must be >= slot_mapping length");
        return ge::GRAPH_FAILED;
    }

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* blockSizePtr = attrs->GetInt(BLOCK_SIZE_ATTR_INDEX);
    const int64_t* modePtr = attrs->GetInt(MODE_ATTR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, blockSizePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, modePtr);
    if (*blockSizePtr <= 0 || (*modePtr != SERIAL_MODE && *modePtr != MULTI_CORE_MODE)) {
        OP_LOGE(context->GetNodeName(), "blockSize must be positive and mode must be 0 or 1");
        return ge::GRAPH_FAILED;
    }

    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    if (aivNum == 0) {
        OP_LOGE(context->GetNodeName(), "failed to get AIV core count");
        return ge::GRAPH_FAILED;
    }

    uint32_t blockDim = 1;
    if (*modePtr == MULTI_CORE_MODE && numSlots > 0) {
        // Scalar SetValue writes through the per-core data cache.  Assign one
        // whole 64-byte output cache line (16 int32 values) to one AIV so two
        // cores never dirty different words of the same cache line.
        uint32_t outputLineCount =
            (outputCapacity + INT32_PER_CACHE_LINE - 1) / INT32_PER_CACHE_LINE;
        blockDim = std::min(aivNum, outputLineCount);
    }

    StoreKvBlockMetadataAivTilingData tilingData;
    tilingData.set_numSlots(numSlots);
    tilingData.set_outputCapacity(outputCapacity);
    tilingData.set_blockSize(static_cast<uint32_t>(*blockSizePtr));
    tilingData.set_mode(static_cast<uint32_t>(*modePtr));
    tilingData.set_usedCoreNum(blockDim);

    size_t* workspaceSize = context->GetWorkspaceSizes(1);
    *workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    context->SetTilingKey(static_cast<uint64_t>(*modePtr + 1));
    context->SetBlockDim(blockDim);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForStoreKvBlockMetadataAiv(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(StoreKvBlockMetadataAiv)
    .Tiling(StoreKvBlockMetadataAivTilingFunc)
    .TilingParse<StoreKvBlockMetadataAivCompileInfo>(TilingParseForStoreKvBlockMetadataAiv);
}  // namespace optiling
