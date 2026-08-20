import os, sys
sys.path.insert(0, "/home/z00980808/zrr_dev/vllm-ascend")
os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0")
import torch, torch_npu
from vllm_ascend.utils import enable_custom_op
enable_custom_op()
ca = torch.ops._C_ascend

torch.manual_seed(2026)
dev = torch.device("npu")

def causal_topk(T, sparse, dev):
    idx = torch.full((T, 1, sparse), -1, dtype=torch.int32, device=dev)
    for t in range(T):
        idx[t, 0, : t + 1] = torch.arange(t + 1, dtype=torch.int32, device=dev)
    return idx

def run_case(name, dtype, T, H, kv_lora, rope_hd, sparse, blk, return_lse=False):
    num_blk = (T + blk - 1) // blk
    ql = torch.randn(T, H, kv_lora, dtype=dtype, device=dev)
    qpe = torch.randn(T, H, rope_hd, dtype=dtype, device=dev)
    kv = torch.randn(num_blk, blk, 1, kv_lora, dtype=dtype, device=dev)
    krope = torch.randn(num_blk, blk, 1, rope_hd, dtype=dtype, device=dev)
    topk = causal_topk(T, sparse, dev)
    bt = torch.arange(num_blk, dtype=torch.int32, device=dev).view(1, num_blk)
    sq = torch.tensor([T], dtype=torch.int32, device=dev)
    sk = torch.tensor([T], dtype=torch.int32, device=dev)
    scale = 1.0 / (kv_lora + rope_hd) ** 0.5

    common = dict(query=ql, key=kv, value=kv, sparse_indices=topk,
                  scale_value=scale, sparse_block_size=1, block_table=bt,
                  actual_seq_lengths_query=sq, actual_seq_lengths_kv=sk,
                  query_rope=qpe, key_rope=krope,
                  layout_query="TND", layout_kv="PA_BSND", sparse_mode=3)
    out_c = ca.npu_sparse_flash_attention(**common, attention_mode=2, return_softmax_lse=return_lse)
    out_n = torch_npu.npu_sparse_flash_attention(**common, attention_mode=2, return_softmax_lse=return_lse)

    a = out_c[0] if isinstance(out_c, tuple) else out_c
    b = out_n[0] if isinstance(out_n, tuple) else out_n
    diff = (a.float() - b.float()).abs()
    ok = torch.allclose(a.float(), b.float(), rtol=1e-2, atol=1e-2)
    line = f"{name:24s} diff_max={diff.max().item():.3e} diff_mean={diff.mean().item():.3e} allclose={ok}"
    if return_lse:
        dmax = (out_c[1] - out_n[1]).abs().max().item()
        dsum = (out_c[2] - out_n[2]).abs().max().item()
        line += f"  lse_max_diff={dmax:.3e} lse_sum_diff={dsum:.3e}"
    print(line)

run_case("bf16_decode_T128", torch.bfloat16, 128, 8, 512, 64, 2048, 128)
run_case("fp16_decode_T128", torch.float16, 128, 8, 512, 64, 2048, 128)
run_case("bf16_prefill_T256", torch.bfloat16, 256, 8, 512, 64, 2048, 128)
run_case("bf16_long_T1024", torch.bfloat16, 1024, 8, 512, 64, 2048, 128)
print("DONE")
