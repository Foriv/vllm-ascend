# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare AICPU, single-AIV and multi-AIV StoreKVBlock metadata kernels."""

import argparse
import statistics
import time
from collections.abc import Callable, Sequence

import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import enable_custom_op

PROVIDERS = {
    "aicpu": "store_kv_block_metadata",
    "aiv_serial": "store_kv_block_metadata_aiv_serial",
    "aiv_multi": "store_kv_block_metadata_aiv_multi",
}


def _csv_ints(value: str) -> list[int]:
    values = [int(item) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def _make_slots(num_tokens: int, block_size: int, pattern: str) -> list[int]:
    start = max(1, block_size // 3)
    if pattern == "contiguous":
        return [start + index for index in range(num_tokens)]
    if pattern == "fragmented":
        return [start + 2 * index for index in range(num_tokens)]
    if pattern == "mixed":
        return [-1 if index % 8 == 0 else start + index for index in range(num_tokens)]
    raise ValueError(f"unknown pattern: {pattern}")


def _new_metadata(slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.empty_like(slots), torch.empty_like(slots), torch.empty_like(slots)


def _runner(
    provider: str,
    slots: torch.Tensor,
    metadata: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    block_size: int,
) -> Callable[[], None]:
    op = getattr(torch.ops._C_ascend, PROVIDERS[provider])
    return lambda: op(slots, *metadata, block_size)


def _active_groups(metadata: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> list[tuple[int, int, int]]:
    group_len, group_key_idx, group_key_cache_idx = (tensor.cpu().tolist() for tensor in metadata)
    return sorted(
        (key_idx, cache_idx, length)
        for length, key_idx, cache_idx in zip(group_len, group_key_idx, group_key_cache_idx)
        if length > 0
    )


def _quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * q)]


def _device_times(
    run: Callable[[], None], warmup: int, samples: int, inner_repeats: int
) -> tuple[float, float, float]:
    for _ in range(warmup):
        run()
    torch.npu.synchronize()

    times: list[float] = []
    for _ in range(samples):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(inner_repeats):
            run()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000.0 / inner_repeats)
    return statistics.median(times), _quantile(times, 0.2), _quantile(times, 0.8)


def _host_device_time(run: Callable[[], None], samples: int, inner_repeats: int) -> float:
    times: list[float] = []
    for _ in range(samples):
        torch.npu.synchronize()
        start = time.perf_counter()
        for _ in range(inner_repeats):
            run()
        torch.npu.synchronize()
        times.append((time.perf_counter() - start) * 1_000_000.0 / inner_repeats)
    return statistics.median(times)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-tokens", type=_csv_ints, default=[16, 128, 512, 2048, 8192])
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--patterns", default="contiguous,fragmented,mixed")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--inner-repeats", type=int, default=20)
    args = parser.parse_args()

    patterns = [item.strip() for item in args.patterns.split(",") if item.strip()]
    if any(pattern not in {"contiguous", "fragmented", "mixed"} for pattern in patterns):
        raise ValueError("patterns must be contiguous, fragmented, or mixed")

    enable_custom_op()
    torch.npu.set_device(0)
    print("tokens pattern     provider     device_us  p20_us  p80_us  host+device_us")
    for num_tokens in args.num_tokens:
        for pattern in patterns:
            slots = torch.tensor(
                _make_slots(num_tokens, args.block_size, pattern),
                dtype=torch.int32,
                device="npu:0",
            )
            states = {provider: _new_metadata(slots) for provider in PROVIDERS}
            runs = {
                provider: _runner(provider, slots, states[provider], args.block_size)
                for provider in PROVIDERS
            }

            for run in runs.values():
                run()
            torch.npu.synchronize()
            reference = _active_groups(states["aicpu"])
            assert states["aicpu"][0].equal(states["aiv_serial"][0])
            assert states["aicpu"][1].equal(states["aiv_serial"][1])
            assert states["aicpu"][2].equal(states["aiv_serial"][2])
            assert reference == _active_groups(states["aiv_multi"])

            for provider, run in runs.items():
                median, p20, p80 = _device_times(
                    run, args.warmup, args.samples, args.inner_repeats
                )
                host_device = _host_device_time(run, args.samples, args.inner_repeats)
                print(
                    f"{num_tokens:6d} {pattern:11s} {provider:12s} "
                    f"{median:9.3f} {p20:7.3f} {p80:7.3f} {host_device:14.3f}"
                )


if __name__ == "__main__":
    main()
