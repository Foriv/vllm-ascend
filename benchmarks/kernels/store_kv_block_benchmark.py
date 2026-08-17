# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Benchmark Ascend C/AICPU and Triton implementations of StoreKVBlock.

This script reports two timing scopes:

* device_us: NPU stream time measured with ``torch.npu.Event``;
* host_device_us: wall-clock launch plus device time, synchronized per sample.

Triton JIT compilation and native-op lazy loading happen during correctness
checks and warmup, before steady-state timing starts.
"""

import argparse
import csv
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import torch_npu  # noqa: F401  # Registers the torch.npu backend.

from vllm_ascend.ops.triton.store_kv_block import (
    store_kv_block_metadata_triton_multi_aiv,
    store_kv_block_metadata_triton_parallel,
    store_kv_block_metadata_triton_serial,
    store_kv_block_triton,
)
from vllm_ascend.utils import enable_custom_op

PROVIDERS = ("ascendc", "triton_serial", "triton_parallel", "triton_multi_aiv")
STAGES = ("metadata", "copy", "e2e")
DTYPES = {
    "int8": torch.int8,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass(frozen=True)
class Workload:
    num_tokens: int
    num_heads: int
    head_size: int
    block_size: int
    num_blocks: int
    pattern: str
    dtype_name: str

    @property
    def token_size(self) -> int:
        return self.num_heads * self.head_size


@dataclass
class ProviderState:
    group_len: torch.Tensor
    group_key_idx: torch.Tensor
    group_key_cache_idx: torch.Tensor
    key_cache: torch.Tensor


@dataclass(frozen=True)
class TimingResult:
    workload: Workload
    stage: str
    provider: str
    device_us: float
    device_p20_us: float
    device_p80_us: float
    host_device_us: float


def _parse_csv_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected a comma-separated list of positive integers")
    return values


def _parse_csv_choices(value: str, choices: Sequence[str]) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    invalid = sorted(set(values) - set(choices))
    if not values or invalid:
        raise argparse.ArgumentTypeError(f"expected comma-separated values from {tuple(choices)}, got {invalid}")
    return values


def _make_slot_mapping(num_tokens: int, block_size: int, pattern: str) -> list[int]:
    start_slot = max(1, block_size // 3)
    if pattern == "contiguous":
        return [start_slot + token_idx for token_idx in range(num_tokens)]
    if pattern == "fragmented":
        return [start_slot + 2 * token_idx for token_idx in range(num_tokens)]
    if pattern == "mixed":
        return [-1 if token_idx % 8 == 0 else start_slot + token_idx for token_idx in range(num_tokens)]
    raise ValueError(f"unsupported slot pattern: {pattern}")


def _required_num_blocks(slot_mapping: Sequence[int], block_size: int) -> int:
    max_slot = max((slot for slot in slot_mapping if slot >= 0), default=-1)
    return max(1, (max_slot + block_size) // block_size)


def _make_random_tensor(shape: tuple[int, ...], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if dtype == torch.int8:
        return torch.randint(-8, 8, shape, dtype=dtype, device=device)
    return torch.randn(shape, dtype=dtype, device=device)


def _new_provider_state(base_cache: torch.Tensor, num_tokens: int) -> ProviderState:
    return ProviderState(
        group_len=torch.empty(num_tokens, dtype=torch.int32, device=base_cache.device),
        group_key_idx=torch.empty(num_tokens, dtype=torch.int32, device=base_cache.device),
        group_key_cache_idx=torch.empty(num_tokens, dtype=torch.int32, device=base_cache.device),
        key_cache=base_cache.clone(),
    )


def _run_metadata(
    provider: str,
    slot_mapping: torch.Tensor,
    state: ProviderState,
    block_size: int,
) -> None:
    if provider == "ascendc":
        torch.ops._C_ascend.store_kv_block_metadata(
            slot_mapping,
            state.group_len,
            state.group_key_idx,
            state.group_key_cache_idx,
            block_size,
        )
        return
    metadata_impl = {
        "triton_serial": store_kv_block_metadata_triton_serial,
        "triton_parallel": store_kv_block_metadata_triton_parallel,
        "triton_multi_aiv": store_kv_block_metadata_triton_multi_aiv,
    }[provider]
    metadata_impl(
        slot_mapping,
        state.group_len,
        state.group_key_idx,
        state.group_key_cache_idx,
        block_size,
    )


def _run_copy(
    provider: str,
    key: torch.Tensor,
    state: ProviderState,
    block_size: int,
) -> None:
    if provider == "ascendc":
        torch.ops._C_ascend.store_kv_block(
            key,
            state.key_cache,
            state.group_len,
            state.group_key_idx,
            state.group_key_cache_idx,
            block_size,
        )
        return
    store_kv_block_triton(
        key,
        state.key_cache,
        state.group_len,
        state.group_key_idx,
        state.group_key_cache_idx,
        block_size,
    )


def _make_runner(
    provider: str,
    stage: str,
    key: torch.Tensor,
    slot_mapping: torch.Tensor,
    state: ProviderState,
    block_size: int,
) -> Callable[[], None]:
    if stage == "metadata":
        return lambda: _run_metadata(provider, slot_mapping, state, block_size)
    if stage == "copy":
        return lambda: _run_copy(provider, key, state, block_size)
    if stage == "e2e":

        def run_e2e() -> None:
            _run_metadata(provider, slot_mapping, state, block_size)
            _run_copy(provider, key, state, block_size)

        return run_e2e
    raise ValueError(f"unsupported stage: {stage}")


def _quantile(samples: Sequence[float], quantile: float) -> float:
    ordered = sorted(samples)
    index = round((len(ordered) - 1) * quantile)
    return ordered[index]


def _measure_device_us(
    runner: Callable[[], None],
    warmup: int,
    samples: int,
    inner_repeats: int,
) -> tuple[float, float, float]:
    for _ in range(warmup):
        runner()
    torch.npu.synchronize()

    durations_us: list[float] = []
    for _ in range(samples):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(inner_repeats):
            runner()
        end.record()
        end.synchronize()
        durations_us.append(start.elapsed_time(end) * 1000.0 / inner_repeats)

    return (
        statistics.median(durations_us),
        _quantile(durations_us, 0.2),
        _quantile(durations_us, 0.8),
    )


def _measure_host_device_us(
    runner: Callable[[], None],
    samples: int,
    inner_repeats: int,
) -> float:
    durations_us: list[float] = []
    for _ in range(samples):
        torch.npu.synchronize()
        start = time.perf_counter()
        for _ in range(inner_repeats):
            runner()
        torch.npu.synchronize()
        durations_us.append((time.perf_counter() - start) * 1_000_000.0 / inner_repeats)
    return statistics.median(durations_us)


def _assert_correctness(
    key: torch.Tensor,
    base_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
    providers: Sequence[str],
) -> None:
    states = {provider: _new_provider_state(base_cache, slot_mapping.numel()) for provider in providers}
    for provider, state in states.items():
        _run_metadata(provider, slot_mapping, state, block_size)
        _run_copy(provider, key, state, block_size)
    torch.npu.synchronize()

    native = states["ascendc"]
    for provider, state in states.items():
        if provider == "ascendc":
            continue
        if provider != "triton_multi_aiv":
            torch.testing.assert_close(native.group_len, state.group_len, rtol=0, atol=0)
            torch.testing.assert_close(native.group_key_idx, state.group_key_idx, rtol=0, atol=0)
            torch.testing.assert_close(native.group_key_cache_idx, state.group_key_cache_idx, rtol=0, atol=0)
        torch.testing.assert_close(native.key_cache, state.key_cache, rtol=0, atol=0)


def _prepare_copy_metadata(
    slot_mapping: torch.Tensor,
    states: dict[str, ProviderState],
    block_size: int,
) -> None:
    # Generate identical compact metadata once, outside copy-only timing.
    reference = states["ascendc"]
    _run_metadata("ascendc", slot_mapping, reference, block_size)
    for provider, state in states.items():
        if provider == "ascendc":
            continue
        state.group_len.copy_(reference.group_len)
        state.group_key_idx.copy_(reference.group_key_idx)
        state.group_key_cache_idx.copy_(reference.group_key_cache_idx)
    torch.npu.synchronize()


def _benchmark_workload(
    workload: Workload,
    stages: Sequence[str],
    providers: Sequence[str],
    warmup: int,
    samples: int,
    inner_repeats: int,
    device: torch.device,
) -> list[TimingResult]:
    dtype = DTYPES[workload.dtype_name]
    slots = _make_slot_mapping(workload.num_tokens, workload.block_size, workload.pattern)
    slot_mapping = torch.tensor(slots, dtype=torch.int32, device=device)
    key = _make_random_tensor(
        (workload.num_tokens, workload.num_heads, workload.head_size),
        dtype,
        device,
    )
    base_cache = _make_random_tensor(
        (
            workload.num_blocks,
            workload.block_size,
            workload.num_heads,
            workload.head_size,
        ),
        dtype,
        device,
    )

    _assert_correctness(key, base_cache, slot_mapping, workload.block_size, providers)

    states = {provider: _new_provider_state(base_cache, workload.num_tokens) for provider in providers}
    if "copy" in stages:
        _prepare_copy_metadata(slot_mapping, states, workload.block_size)

    results: list[TimingResult] = []
    for stage in stages:
        for provider in providers:
            runner = _make_runner(
                provider,
                stage,
                key,
                slot_mapping,
                states[provider],
                workload.block_size,
            )
            device_us, p20_us, p80_us = _measure_device_us(
                runner,
                warmup,
                samples,
                inner_repeats,
            )
            host_device_us = _measure_host_device_us(runner, samples, inner_repeats)
            results.append(
                TimingResult(
                    workload=workload,
                    stage=stage,
                    provider=provider,
                    device_us=device_us,
                    device_p20_us=p20_us,
                    device_p80_us=p80_us,
                    host_device_us=host_device_us,
                )
            )
    return results


def _result_row(result: TimingResult) -> dict[str, object]:
    workload = result.workload
    return {
        "num_tokens": workload.num_tokens,
        "num_heads": workload.num_heads,
        "head_size": workload.head_size,
        "token_size": workload.token_size,
        "block_size": workload.block_size,
        "num_blocks": workload.num_blocks,
        "pattern": workload.pattern,
        "dtype": workload.dtype_name,
        "stage": result.stage,
        "provider": result.provider,
        "device_us": f"{result.device_us:.3f}",
        "device_p20_us": f"{result.device_p20_us:.3f}",
        "device_p80_us": f"{result.device_p80_us:.3f}",
        "host_device_us": f"{result.host_device_us:.3f}",
    }


def _print_results(results: Sequence[TimingResult], selected_providers: Sequence[str]) -> None:
    print("tokens heads head_size pattern    dtype     stage     provider        device_us  p20_us  p80_us  host+device_us")
    for result in results:
        workload = result.workload
        print(
            f"{workload.num_tokens:6d} {workload.num_heads:5d} "
            f"{workload.head_size:9d} {workload.pattern:10s} "
            f"{workload.dtype_name:9s} {result.stage:9s} {result.provider:15s} "
            f"{result.device_us:9.3f} {result.device_p20_us:7.3f} "
            f"{result.device_p80_us:7.3f} {result.host_device_us:14.3f}"
        )

    print("\nDevice-time speedup (ascendc / provider; >1 means provider is faster):")
    grouped: dict[tuple[Workload, str], dict[str, TimingResult]] = {}
    for result in results:
        grouped.setdefault((result.workload, result.stage), {})[result.provider] = result
    for (workload, stage), providers in grouped.items():
        if "ascendc" not in providers:
            continue
        for provider in selected_providers:
            if provider == "ascendc" or provider not in providers:
                continue
            provider_us = providers[provider].device_us
            speedup = providers["ascendc"].device_us / provider_us if provider_us > 0 else float("inf")
            print(
                f"tokens={workload.num_tokens:<6d} head={workload.head_size:<5d} "
                f"pattern={workload.pattern:<10s} stage={stage:<9s} "
                f"provider={provider:<15s} speedup={speedup:.3f}x"
            )


def _write_csv(results: Sequence[TimingResult], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [_result_row(result) for result in results]
    with output.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-tokens", type=_parse_csv_ints, default=[16, 128, 512])
    parser.add_argument("--num-heads", type=int, default=1)
    parser.add_argument("--head-sizes", type=_parse_csv_ints, default=[64])
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=0, help="0 chooses the minimum safe cache size")
    parser.add_argument(
        "--patterns",
        type=lambda value: _parse_csv_choices(value, ("contiguous", "fragmented", "mixed")),
        default=["contiguous", "fragmented", "mixed"],
    )
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="int8")
    parser.add_argument(
        "--stages",
        type=lambda value: _parse_csv_choices(value, STAGES),
        default=list(STAGES),
    )
    parser.add_argument(
        "--providers",
        type=lambda value: _parse_csv_choices(value, PROVIDERS),
        default=list(PROVIDERS),
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--inner-repeats", type=int, default=20)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None, help="optional CSV output path")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.num_heads <= 0 or args.block_size <= 0:
        raise ValueError("num_heads and block_size must be positive")
    if args.num_blocks < 0:
        raise ValueError("num_blocks must be non-negative")
    if args.warmup < 0 or args.samples <= 0 or args.inner_repeats <= 0:
        raise ValueError("warmup must be non-negative; samples and inner_repeats must be positive")
    if "ascendc" not in args.providers:
        raise ValueError("--providers must include ascendc as the correctness and speedup reference")

    enable_custom_op()
    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    torch.manual_seed(args.seed)

    workloads: list[Workload] = []
    for num_tokens in args.num_tokens:
        for head_size in args.head_sizes:
            for pattern in args.patterns:
                slots = _make_slot_mapping(num_tokens, args.block_size, pattern)
                required_blocks = _required_num_blocks(slots, args.block_size)
                if args.num_blocks and args.num_blocks < required_blocks:
                    raise ValueError(
                        f"num_blocks={args.num_blocks} is too small for "
                        f"tokens={num_tokens}, pattern={pattern}; need at least {required_blocks}"
                    )
                num_blocks = args.num_blocks or required_blocks
                workloads.append(
                    Workload(
                        num_tokens=num_tokens,
                        num_heads=args.num_heads,
                        head_size=head_size,
                        block_size=args.block_size,
                        num_blocks=num_blocks,
                        pattern=pattern,
                        dtype_name=args.dtype,
                    )
                )

    all_results: list[TimingResult] = []
    with torch.inference_mode():
        for workload in workloads:
            print(f"Benchmarking {workload} ...", flush=True)
            all_results.extend(
                _benchmark_workload(
                    workload,
                    args.stages,
                    args.providers,
                    args.warmup,
                    args.samples,
                    args.inner_repeats,
                    device,
                )
            )

    _print_results(all_results, args.providers)
    if args.output is not None:
        _write_csv(all_results, args.output)
        print(f"\nCSV written to {args.output}")


if __name__ == "__main__":
    main()
