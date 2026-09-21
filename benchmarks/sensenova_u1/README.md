# SenseNova-U1 mixed-traffic serving benchmark

This benchmark measures the sequential mixed workload relevant to startup
warmup, regional compilation, and autoregressive CUDA graph capture. It sends
no unrecorded client-side warmup requests: cycle 0 includes the first requests
after server readiness, while later cycles measure reuse.

Start a fresh server for every configuration. For example, an eager baseline:

```bash
vllm serve sensenova/SenseNova-U1.5-8B-MoT --omni --port 8091 \
  --enforce-eager
```

Run the benchmark:

```bash
python benchmarks/sensenova_u1/benchmark_mixed_serving.py \
  --server http://localhost:8091 \
  --model sensenova/SenseNova-U1.5-8B-MoT \
  --cycles 3 \
  --server-configuration eager \
  --output-file results/sensenova-eager.json
```

Then restart the server with regional dynamic compilation:

```bash
vllm serve sensenova/SenseNova-U1.5-8B-MoT --omni --port 8091 \
  --diffusion-compile-granularity regional \
  --diffusion-compile-dynamic

python benchmarks/sensenova_u1/benchmark_mixed_serving.py \
  --server http://localhost:8091 \
  --model sensenova/SenseNova-U1.5-8B-MoT \
  --cycles 3 \
  --server-configuration regional-dynamic \
  --output-file results/sensenova-regional-dynamic.json
```

The built-in sequence covers T2I with think disabled and enabled, T2T, I2T,
I2I, a second image resolution, and repeated T2I/T2T shapes. Results contain
every request latency plus overall, per-cycle, and per-shape-group P50/P100.
They also record the client-side commit SHA, dirty-worktree state, GPU details,
request sequence, and benchmark configuration.

Use the same commit, prompts, image, seed, request sequence, and server settings
for comparisons. Run each configuration in a fresh process. For a true cold
compile comparison, give each server run an isolated Torch/Inductor cache.
Server startup-to-ready time and server-side compile/CUDA-graph counters cannot
be inferred by this external client; record them separately from server logs.

## Custom sequence

Pass `--sequence-file sequence.json`. The file is a JSON list:

```json
[
  {
    "name": "t2i_square",
    "modality": "text2img",
    "prompt": "A red apple on a wooden table",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 8,
    "cfg_scale": 4.0
  },
  {
    "name": "t2i_square_repeat",
    "group": "t2i_square",
    "modality": "text2img",
    "width": 1024,
    "height": 1024
  },
  {
    "name": "understanding",
    "modality": "img2text",
    "uses_image": true,
    "max_tokens": 64
  }
]
```

`group` combines intentional repeats under one per-shape summary. Supported
modalities are `text2img`, `img2img`, `text2text`, and `img2text`.
