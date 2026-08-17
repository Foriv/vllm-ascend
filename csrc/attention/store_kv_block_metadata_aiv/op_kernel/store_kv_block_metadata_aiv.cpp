/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include "store_kv_block_metadata_aiv.h"

extern "C" __global__ __aicore__ void store_kv_block_metadata_aiv(
    GM_ADDR slotMapping, GM_ADDR groupLen, GM_ADDR groupKeyIdx, GM_ADDR groupKeyCacheIdx,
    GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    REGISTER_TILING_DEFAULT(StoreKvBlockMetadataAiv::StoreKvBlockMetadataAivTilingData);
    GET_TILING_DATA(tilingData, tiling);

    StoreKvBlockMetadataAiv::StoreKvBlockMetadataAivKernel op;
    op.Init(slotMapping, groupLen, groupKeyIdx, groupKeyCacheIdx, &tilingData);
    if (TILING_KEY_IS(1)) {
        op.ProcessSerial();
    } else if (TILING_KEY_IS(2)) {
        op.ProcessMultiCore();
    }
}

