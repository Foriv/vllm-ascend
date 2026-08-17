# Kernel microbenchmarks

## StoreKVBlock

`store_kv_block_benchmark.py` compares the existing native implementation
(AICPU metadata + Ascend C copy) with the Triton implementation.

Run a quick smoke benchmark inside the A2 development container:

```bash
cd /home/z00980808/zrr_dev/vllm-ascend
export PYTHONPATH=$PWD:$PYTHONPATH

python benchmarks/kernels/store_kv_block_benchmark.py \
  --num-tokens 16,128 \
  --head-sizes 64 \
  --patterns contiguous,fragmented,mixed \
  --dtype int8 \
  --warmup 5 \
  --samples 10 \
  --inner-repeats 10
```

Run the fuller comparison and save raw rows:

```bash
python benchmarks/kernels/store_kv_block_benchmark.py \
  --num-tokens 16,128,512,2048 \
  --head-sizes 64,512 \
  --patterns contiguous,fragmented,mixed \
  --dtype int8 \
  --stages metadata,copy,e2e \
  --warmup 10 \
  --samples 30 \
  --inner-repeats 20 \
  --output store_kv_block_benchmark.csv
```

Metrics:

- `device_us`: median NPU stream time per operator call, measured with NPU
  events. Triton JIT and native lazy loading are excluded by correctness and
  warmup runs.
- `p20_us` / `p80_us`: spread of device-time samples.
- `host+device_us`: synchronized wall-clock time per call, including Python,
  dispatch, launch, and device execution.
- `speedup = ascendc / triton`: values greater than 1 mean Triton is faster.

Stages:

- `metadata`: AICPU metadata versus Triton metadata only.
- `copy`: both providers consume the same precomputed metadata, so this isolates
  Ascend C copy versus Triton copy.
- `e2e`: metadata followed by copy for each provider.

The script verifies metadata and final cache equality before measuring every
workload. It intentionally keeps first-JIT latency out of steady-state results;
measure first-call latency separately when deployment startup matters.
