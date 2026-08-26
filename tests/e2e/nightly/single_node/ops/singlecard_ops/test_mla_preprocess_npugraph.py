import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op


torch.npu.config.allow_internal_format = True
enable_custom_op()


def _build_case() -> dict[str, object]:
    torch.manual_seed(0)
    token_num = 2
    head_num = 2
    hidden_size = 7168
    mm1_out = 2112
    q_lora_rank = 1536
    kv_lora_rank = 512
    rope_dim = 64
    block_num = 2
    block_size = 128
    dtype = torch.bfloat16

    wdqkv = torch.randint(0, 7, (1, hidden_size // 32, mm1_out, 32), dtype=torch.int8, device="npu")
    wdqkv = torch_npu.npu_format_cast(wdqkv.contiguous(), 29)
    wuq = torch.randint(0, 7, (1, q_lora_rank // 32, head_num * 192, 32), dtype=torch.int8, device="npu")
    wuq = torch_npu.npu_format_cast(wuq.contiguous(), 29)
    wuk = torch.randn((head_num, 128, kv_lora_rank), dtype=dtype, device="npu")
    wuk = torch_npu.npu_format_cast(wuk, 29)

    return {
        "hidden_states": torch.randn((token_num, hidden_size), dtype=dtype, device="npu"),
        "wdqkv": wdqkv,
        "de_scale0": torch.rand((mm1_out,), dtype=torch.float32, device="npu"),
        "gamma1": torch.randn((q_lora_rank,), dtype=dtype, device="npu"),
        "beta1": torch.randn((q_lora_rank,), dtype=dtype, device="npu"),
        "wuq": wuq,
        "de_scale1": torch.rand((head_num * 192,), dtype=torch.float32, device="npu"),
        "gamma2": torch.randn((kv_lora_rank,), dtype=dtype, device="npu"),
        "cos": torch.randn((token_num, rope_dim), dtype=dtype, device="npu"),
        "sin": torch.randn((token_num, rope_dim), dtype=dtype, device="npu"),
        "wuk": wuk,
        "slotmapping": torch.arange(token_num, dtype=torch.int32, device="npu"),
        "quant_scale0": torch.tensor([0.25], dtype=dtype, device="npu"),
        "quant_offset0": torch.zeros((1,), dtype=torch.int8, device="npu"),
        "bias0": torch.randint(0, 7, (mm1_out,), dtype=torch.int32, device="npu"),
        "quant_scale1": torch.tensor([0.25], dtype=dtype, device="npu"),
        "quant_offset1": torch.zeros((1,), dtype=torch.int8, device="npu"),
        "bias1": torch.randint(0, 7, (head_num * 192,), dtype=torch.int32, device="npu"),
        "ctkv_scale": torch.ones((1,), dtype=dtype, device="npu"),
        "q_nope_scale": torch.ones((head_num,), dtype=dtype, device="npu"),
        "token_num": token_num,
        "head_num": head_num,
        "q_lora_rank": q_lora_rank,
        "kv_lora_rank": kv_lora_rank,
        "rope_dim": rope_dim,
        "block_num": block_num,
        "block_size": block_size,
        "dtype": dtype,
    }


def _new_outputs(case: dict[str, object]) -> dict[str, torch.Tensor]:
    token_num = int(case["token_num"])
    head_num = int(case["head_num"])
    q_lora_rank = int(case["q_lora_rank"])
    kv_lora_rank = int(case["kv_lora_rank"])
    rope_dim = int(case["rope_dim"])
    block_num = int(case["block_num"])
    block_size = int(case["block_size"])
    dtype = case["dtype"]
    assert isinstance(dtype, torch.dtype)
    return {
        "q_nope": torch.zeros((token_num, head_num, kv_lora_rank), dtype=dtype, device="npu"),
        "q_rope": torch.zeros((token_num, head_num, rope_dim), dtype=dtype, device="npu"),
        "inner": torch.zeros((token_num, q_lora_rank), dtype=dtype, device="npu"),
        "kv_cache": torch.zeros((block_num, block_size, kv_lora_rank), dtype=dtype, device="npu"),
        "rope_cache": torch.zeros((block_num, block_size, rope_dim), dtype=dtype, device="npu"),
    }


def _run(case: dict[str, object], outputs: dict[str, torch.Tensor]) -> None:
    torch.ops._C_ascend.mla_preprocess(
        case["hidden_states"],
        case["wdqkv"],
        case["de_scale0"],
        case["gamma1"],
        case["beta1"],
        case["wuq"],
        case["de_scale1"],
        case["gamma2"],
        case["cos"],
        case["sin"],
        case["wuk"],
        outputs["kv_cache"],
        outputs["rope_cache"],
        case["slotmapping"],
        quant_scale0=case["quant_scale0"],
        quant_offset0=case["quant_offset0"],
        bias0=case["bias0"],
        quant_scale1=case["quant_scale1"],
        quant_offset1=case["quant_offset1"],
        bias1=case["bias1"],
        ctkv_scale=case["ctkv_scale"],
        q_nope_scale=case["q_nope_scale"],
        cache_mode="krope_ctkv",
        quant_mode="per_tensor_quant_asymm",
        enable_inner_out=True,
        q_out0=outputs["q_nope"],
        kv_cache_out0=outputs["kv_cache"],
        q_out1=outputs["q_rope"],
        kv_cache_out1=outputs["rope_cache"],
        inner_out=outputs["inner"],
    )


def _assert_outputs(actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]) -> None:
    for name in ("q_nope", "q_rope", "inner", "kv_cache", "rope_cache"):
        torch.testing.assert_close(actual[name], expected[name], rtol=2e-2, atol=2e-2)


@torch.inference_mode()
def test_mla_preprocess_npugraph_capture_and_replay():
    case = _build_case()
    eager_outputs = _new_outputs(case)
    graph_outputs = _new_outputs(case)

    # This is the production SFA signature and also warms the ACLNN executor.
    _run(case, eager_outputs)
    torch.npu.synchronize()
    for name in ("q_nope", "q_rope", "inner"):
        assert torch.isfinite(eager_outputs[name]).all(), f"eager {name} contains NaN/Inf"

    torch.npu.empty_cache()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        _run(case, graph_outputs)

    graph.replay()
    torch.npu.synchronize()
    _assert_outputs(graph_outputs, eager_outputs)
    first_q_nope = graph_outputs["q_nope"].clone()

    hidden_states = case["hidden_states"]
    cos = case["cos"]
    sin = case["sin"]
    assert isinstance(hidden_states, torch.Tensor)
    assert isinstance(cos, torch.Tensor)
    assert isinstance(sin, torch.Tensor)
    hidden_states.copy_(torch.randn_like(hidden_states))
    cos.copy_(torch.randn_like(cos))
    sin.copy_(torch.randn_like(sin))

    replay_expected = _new_outputs(case)
    _run(case, replay_expected)
    for tensor in graph_outputs.values():
        tensor.zero_()
    graph.replay()
    torch.npu.synchronize()

    _assert_outputs(graph_outputs, replay_expected)
    assert not torch.allclose(graph_outputs["q_nope"], first_q_nope), "replay reused stale input values"
