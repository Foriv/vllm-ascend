/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MLA_PREPROCESS_HOST_TILING_DATA_H
#define MLA_PREPROCESS_HOST_TILING_DATA_H

#include "register/tilingdata_base.h"

namespace optiling {

// Keep this registration flat. TILING_DATA_FIELD_DEF_STRUCT aligns every
// nested object to eight bytes, while the device PpMatmulTilingData objects
// have four-byte alignment. A flat registration therefore preserves the
// exact natural C++ layout consumed by the AI Core kernel.
BEGIN_TILING_DATA_DEF(MlaPreprocessTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, tilingKey);
    TILING_DATA_FIELD_DEF(uint64_t, userWorkspaceSize);
    TILING_DATA_FIELD_DEF(uint64_t, s1Offset);
    TILING_DATA_FIELD_DEF(uint64_t, s2Offset);
    TILING_DATA_FIELD_DEF(uint64_t, s3Offset);
    TILING_DATA_FIELD_DEF(uint64_t, s4Offset);
    TILING_DATA_FIELD_DEF(uint64_t, s5Offset);

    TILING_DATA_FIELD_DEF(uint32_t, numCore);
    TILING_DATA_FIELD_DEF(uint32_t, n);
    TILING_DATA_FIELD_DEF(uint32_t, perTaskNum);
    TILING_DATA_FIELD_DEF(uint32_t, resTaskNum);

    TILING_DATA_FIELD_DEF(uint32_t, mm1NumBatch);
    TILING_DATA_FIELD_DEF(uint32_t, mm1M);
    TILING_DATA_FIELD_DEF(uint32_t, mm1K);
    TILING_DATA_FIELD_DEF(uint32_t, mm1N);
    TILING_DATA_FIELD_DEF(uint32_t, mm1M0);
    TILING_DATA_FIELD_DEF(uint32_t, mm1K0);
    TILING_DATA_FIELD_DEF(uint32_t, mm1N0);
    TILING_DATA_FIELD_DEF(uint32_t, mm1MLoop);
    TILING_DATA_FIELD_DEF(uint32_t, mm1KLoop);
    TILING_DATA_FIELD_DEF(uint32_t, mm1NLoop);
    TILING_DATA_FIELD_DEF(uint32_t, mm1CoreLoop);
    TILING_DATA_FIELD_DEF(uint32_t, mm1SwizzleCount);
    TILING_DATA_FIELD_DEF(uint32_t, mm1SwizzleDirect);
    TILING_DATA_FIELD_DEF(uint32_t, mm1EnShuffleK);
    TILING_DATA_FIELD_DEF(uint32_t, mm1BlockDim);
    TILING_DATA_FIELD_DEF(uint32_t, mm1EnLoadAllAmat);
    TILING_DATA_FIELD_DEF(uint32_t, mm1B0matPingPongBufferLen);

    TILING_DATA_FIELD_DEF(uint32_t, mm2NumBatch);
    TILING_DATA_FIELD_DEF(uint32_t, mm2M);
    TILING_DATA_FIELD_DEF(uint32_t, mm2K);
    TILING_DATA_FIELD_DEF(uint32_t, mm2N);
    TILING_DATA_FIELD_DEF(uint32_t, mm2M0);
    TILING_DATA_FIELD_DEF(uint32_t, mm2K0);
    TILING_DATA_FIELD_DEF(uint32_t, mm2N0);
    TILING_DATA_FIELD_DEF(uint32_t, mm2MLoop);
    TILING_DATA_FIELD_DEF(uint32_t, mm2KLoop);
    TILING_DATA_FIELD_DEF(uint32_t, mm2NLoop);
    TILING_DATA_FIELD_DEF(uint32_t, mm2CoreLoop);
    TILING_DATA_FIELD_DEF(uint32_t, mm2SwizzleCount);
    TILING_DATA_FIELD_DEF(uint32_t, mm2SwizzleDirect);
    TILING_DATA_FIELD_DEF(uint32_t, mm2EnShuffleK);
    TILING_DATA_FIELD_DEF(uint32_t, mm2BlockDim);
    TILING_DATA_FIELD_DEF(uint32_t, mm2EnLoadAllAmat);
    TILING_DATA_FIELD_DEF(uint32_t, mm2B0matPingPongBufferLen);

    TILING_DATA_FIELD_DEF(uint32_t, mm3NumBatch);
    TILING_DATA_FIELD_DEF(uint32_t, mm3M);
    TILING_DATA_FIELD_DEF(uint32_t, mm3K);
    TILING_DATA_FIELD_DEF(uint32_t, mm3N);
    TILING_DATA_FIELD_DEF(uint32_t, mm3M0);
    TILING_DATA_FIELD_DEF(uint32_t, mm3K0);
    TILING_DATA_FIELD_DEF(uint32_t, mm3N0);
    TILING_DATA_FIELD_DEF(uint32_t, mm3MLoop);
    TILING_DATA_FIELD_DEF(uint32_t, mm3KLoop);
    TILING_DATA_FIELD_DEF(uint32_t, mm3NLoop);
    TILING_DATA_FIELD_DEF(uint32_t, mm3CoreLoop);
    TILING_DATA_FIELD_DEF(uint32_t, mm3SwizzleCount);
    TILING_DATA_FIELD_DEF(uint32_t, mm3SwizzleDirect);
    TILING_DATA_FIELD_DEF(uint32_t, mm3EnShuffleK);
    TILING_DATA_FIELD_DEF(uint32_t, mm3BlockDim);
    TILING_DATA_FIELD_DEF(uint32_t, mm3EnLoadAllAmat);
    TILING_DATA_FIELD_DEF(uint32_t, mm3B0matPingPongBufferLen);

    TILING_DATA_FIELD_DEF(uint32_t, rmsNumCore1);
    TILING_DATA_FIELD_DEF(uint32_t, rmsNumCol1);
    TILING_DATA_FIELD_DEF(uint32_t, rmsNumRow1);
    TILING_DATA_FIELD_DEF(uint32_t, rmsQuantMin1);
    TILING_DATA_FIELD_DEF(uint32_t, rmsNumCore2);
    TILING_DATA_FIELD_DEF(uint32_t, rmsNumCol2);
    TILING_DATA_FIELD_DEF(uint32_t, rmsNumRow2);
    TILING_DATA_FIELD_DEF(uint32_t, rmsQuantMin2);

    TILING_DATA_FIELD_DEF(uint32_t, hiddenSizeQ);
    TILING_DATA_FIELD_DEF(uint32_t, headNumQ);
    TILING_DATA_FIELD_DEF(uint32_t, headDim);
    TILING_DATA_FIELD_DEF(uint32_t, concatSize);
    TILING_DATA_FIELD_DEF(uint32_t, rotaryCoeff);
    TILING_DATA_FIELD_DEF(uint32_t, ntokens);
    TILING_DATA_FIELD_DEF(uint32_t, realCore);
    TILING_DATA_FIELD_DEF(uint32_t, nlCoreRun);
    TILING_DATA_FIELD_DEF(uint32_t, lCoreRun);
    TILING_DATA_FIELD_DEF(uint32_t, maxNPerLoopForUb);
    TILING_DATA_FIELD_DEF(uint32_t, preCoreLoopTime);
    TILING_DATA_FIELD_DEF(uint32_t, preCoreLoopNLast);
    TILING_DATA_FIELD_DEF(uint32_t, lastCoreLoopTime);
    TILING_DATA_FIELD_DEF(uint32_t, lastCoreLoopNLast);

    TILING_DATA_FIELD_DEF(uint32_t, esqFrontCore);
    TILING_DATA_FIELD_DEF(uint32_t, esqTailCore);
    TILING_DATA_FIELD_DEF(uint32_t, esqFrontCoreBatch);
    TILING_DATA_FIELD_DEF(uint32_t, esqTailCoreBatch);
    TILING_DATA_FIELD_DEF(uint32_t, esqHeadNum);
    TILING_DATA_FIELD_DEF(uint32_t, esqColNum);
    TILING_DATA_FIELD_DEF(uint32_t, esqUbHeadLoop);
    TILING_DATA_FIELD_DEF(uint32_t, esqHeadPerLoop);
    TILING_DATA_FIELD_DEF(uint32_t, esqHeadTail);
    TILING_DATA_FIELD_DEF(uint32_t, esqColLoop);
    TILING_DATA_FIELD_DEF(uint32_t, esqColTail);

    TILING_DATA_FIELD_DEF(uint32_t, hiddenStateDim);
    TILING_DATA_FIELD_DEF(uint32_t, isWeightQuantized);
    TILING_DATA_FIELD_DEF(uint32_t, enableRope);
    TILING_DATA_FIELD_DEF(uint32_t, mm1OutSize);
    TILING_DATA_FIELD_DEF(uint32_t, splitSizeOne);
    TILING_DATA_FIELD_DEF(uint32_t, splitSizeTwo);
    TILING_DATA_FIELD_DEF(uint32_t, splitRmsNormSizeOne);
    TILING_DATA_FIELD_DEF(uint32_t, splitRmsNormSizeTwo);
    TILING_DATA_FIELD_DEF(uint32_t, ropeSplitSizeOne);
    TILING_DATA_FIELD_DEF(uint32_t, ropeSplitSizeTwo);
    TILING_DATA_FIELD_DEF(uint32_t, hiddenStrideRope);
    TILING_DATA_FIELD_DEF(uint32_t, qkNopeHeadDim);
    TILING_DATA_FIELD_DEF(float, avgFactor);
    TILING_DATA_FIELD_DEF(uint64_t, kvCacheBlockSize);
    TILING_DATA_FIELD_DEF(uint64_t, kvCacheStride0);
    TILING_DATA_FIELD_DEF(uint64_t, kvCacheRopeStride0);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MlaPreprocess, MlaPreprocessTilingData)

}  // namespace optiling

#endif  // MLA_PREPROCESS_HOST_TILING_DATA_H
