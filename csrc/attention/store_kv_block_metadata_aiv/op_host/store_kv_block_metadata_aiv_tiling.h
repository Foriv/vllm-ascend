/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#ifndef STORE_KV_BLOCK_METADATA_AIV_TILING_H
#define STORE_KV_BLOCK_METADATA_AIV_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(StoreKvBlockMetadataAivTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, numSlots);
    TILING_DATA_FIELD_DEF(uint32_t, outputCapacity);
    TILING_DATA_FIELD_DEF(uint32_t, blockSize);
    TILING_DATA_FIELD_DEF(uint32_t, mode);
    TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(StoreKvBlockMetadataAiv, StoreKvBlockMetadataAivTilingData)

struct StoreKvBlockMetadataAivCompileInfo {};
}  // namespace optiling

#endif  // STORE_KV_BLOCK_METADATA_AIV_TILING_H
