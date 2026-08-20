import os, sys
sys.path.insert(0, "/home/z00980808/zrr_dev/vllm-ascend")
os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0")
import torch, torch_npu
from vllm_ascend.utils import enable_custom_op

MODE = sys.argv[1]  # save | cmp
REF = "/home/z00980808/zrr_dev/sfa_routeA_ref.pt"

ok = enable_custom_op()
print(f"[env] enable_custom_op={ok}  ASCEND_CUSTOM_OPP_PATH={os.environ.get('ASCEND_CUSTOM_OPP_PATH')!r}")
ca = torch.ops._C_ascend
dev = torch.device("npu")


def causal_topk(T, sparse, dev):
    idx = torch.full((T, 1, sparse), -1, dtype=torch.int32, device=dev)
    for t in range(T):
        idx[t, 0, : t + 1] = torch.arange(t + 1, dtype=torch.int32, device=dev)
    return idx


def random_topk(T, sparse, kv_len, dev):
    idx = torch.full((T, 1, sparse), -1, dtype=torch.int32, device=dev)
    for t in range(T):
        n = max(1, int(torch.randint(1, min(sparse, kv_len), (1,)).item()))
        perm = torch.randperm(kv_len, device=dev)[:n].to(torch.int32)
        idx[t, 0, :n] = perm.sort().values
    return idx


def golden(ql, qpe, kv, krope, topk, scale):
    kv_flat = kv.reshape(-1, kv.shape[-1]).float()
    kr_flat = krope.reshape(-1, krope.shape[-1]).float()
    idx = topk[:, 0, :].long()
    valid = idx >= 0
    idxc = idx.clamp(min=0)
    k = kv_flat[idxc]
    kr = kr_flat[idxc]
    q = ql.float()
    qp = qpe.float()
    s = torch.einsum('thd,tkd->thk', q, k) + torch.einsum('thd,tkd->thk', qp, kr)
    s = s * scale
    s = s.masked_fill(~valid[:, None, :], float('-inf'))
    m = s.max(-1).values
    p = torch.exp(s - m[..., None])
    l = p.sum(-1)
    o = torch.einsum('thk,tkd->thd', p, k) / l[..., None]
    return o, m, l


def run_case(name, dtype, T, causal=True, H=8, kv_lora=512, rope_hd=64, sparse=2048, blk=128):
    torch.manual_seed(abs(hash(name)) % (2**31))
    num_blk = (T + blk - 1) // blk
    ql = torch.randn(T, H, kv_lora, dtype=dtype, device=dev)
    qpe = torch.randn(T, H, rope_hd, dtype=dtype, device=dev)
    kv = torch.randn(num_blk, blk, 1, kv_lora, dtype=dtype, device=dev)
    krope = torch.randn(num_blk, blk, 1, rope_hd, dtype=dtype, device=dev)
    topk = causal_topk(T, sparse, dev) if causal else random_topk(T, sparse, T, dev)
    bt = torch.arange(num_blk, dtype=torch.int32, device=dev).view(1, num_blk)
    sq = torch.tensor([T], dtype=torch.int32, device=dev)
    sk = torch.tensor([T], dtype=torch.int32, device=dev)
    scale = 1.0 / (kv_lora + rope_hd) ** 0.5
    smode = 3 if causal else 0

    out, smax, ssum = ca.npu_sparse_flash_attention(
        query=ql, key=kv, value=kv, sparse_indices=topk, scale_value=scale,
        sparse_block_size=1, block_table=bt,
        actual_seq_lengths_query=sq, actual_seq_lengths_kv=sk,
        query_rope=qpe, key_rope=krope,
        layout_query="TND", layout_kv="PA_BSND",
        sparse_mode=smode, attention_mode=2, return_softmax_lse=True)
    og, mg, lg = golden(ql, qpe, kv, krope, topk, scale)
    print(f"  [{name}] smax.shape={tuple(smax.shape)} ssum.shape={tuple(ssum.shape)}")
    return dict(out=out.float().cpu(), smax=smax.cpu(), ssum=ssum.cpu(),
                og=og.cpu(), mg=mg.cpu(), lg=lg.cpu())


CASES = [
    ("bf16_causal_T128",  torch.bfloat16, 128,  True),
    ("bf16_causal_T256",  torch.bfloat16, 256,  True),
    ("bf16_causal_T1024", torch.bfloat16, 1024, True),
    ("fp16_causal_T128",  torch.float16,  128,  True),
    ("bf16_dcpmode0_T256", torch.bfloat16, 256, False),  # sparse_mode=0 + 随机 topk（DCP decode 形态）
]

results = {}
for name, dt, T, causal in CASES:
    results[name] = run_case(name, dt, T, causal)

if MODE == "save":
    torch.save(results, REF)
    print("SAVED", REF)
else:
    ref = torch.load(REF)
    print(f"{'case':22s} {'out_vs_cust':>12s} {'smax_vs_cust':>14s} {'ssum_vs_cust':>14s} | {'smax_vs_gold':>13s} {'ssum_vs_gold':>13s} | {'cust_smax_vs_gold':>18s}")
    for name in ref:
        c, r = results[name], ref[name]
        d_out = (c["out"] - r["out"]).abs().max().item()
        d_smax = (c["smax"] - r["smax"]).abs().max().item()
        d_ssum = (c["ssum"] - r["ssum"]).abs().max().item()
        g_smax = (c["smax"] - r["mg"]).abs().max().item()
        g_ssum = (c["ssum"] - r["lg"]).abs().max().item()
        r_smax = (r["smax"] - r["mg"]).abs().max().item()
        print(f"{name:22s} {d_out:12.3e} {d_smax:14.3e} {d_ssum:14.3e} | {g_smax:13.3e} {g_ssum:13.3e} | {r_smax:18.3e}")
print("DONE")
