# KVShuttle

Benchmark and decision framework for KV cache transfer compression in disaggregated LLM serving.

KVShuttle evaluates 14+ compression strategies across multiple models and sequence lengths, providing GPU-calibrated timing data and analytical transfer simulation to help practitioners choose the right compression scheme for their bandwidth regime.

## Key Results

| Strategy | Ratio | Key cos | Val cos | Token Agree | T4 Speedup | A100 Speedup |
|---|---|---|---|---|---|---|
| uniform_int8 | 2.0x | 0.9998 | 0.9998 | 0.996 | 34-80x | 190-456x |
| kivi_2bit | 6.5x | 0.9649 | 0.8706 | 0.823 | 53-68x | 284-373x |
| uniform_int4 | 3.6x | 0.9872 | 0.9934 | 0.687 | 62-63x | 334-311x |
| fp8_e4m3 | 2.0x | — | — | — | 36-64x | 355-456x |
| cachegen | 3.5x | 0.9927 | 0.9854 | 0.979 | 33-56x | 106-275x |
| cascade_prune50_int4 | 7.1x | 0.7370 | 0.7021 | 0.432 | 48-60x | 276-223x |
| palu_lr | 2.4x | 0.9829 | 0.9778 | 0.752 | 5-216x | 8-848x |

*GPU speedups (compress-decompress) over CPU numpy, measured with CUDA Event timing (zero-copy). T4 = Tesla T4 (320 GB/s), A100 = A100-SXM4-40GB (2039 GB/s). Token agreement = greedy decode match rate with original KV cache (3 models, 10 WikiText prompts).*

**Cosine similarity is a poor predictor of generation quality.** Uniform INT4 achieves 0.99 key cosine similarity yet produces a perplexity delta of ~1,760 — nearly as bad as cascade_prune50_int4 (cosine sim 0.74, ppl delta ~1,930). Small per-element quantization errors compound through multi-head attention layers, so high KV cache fidelity does not guarantee faithful generation. End-to-end metrics like perplexity delta and token agreement are essential for evaluating compression schemes.

## Features

- **14+ compressors** — identity, uniform INT8/INT4, FP8, KIVI 2-bit, CacheGen, KVQuant, Palu (SVD), token pruning, cascade (prune+quantize), lossless (zstd/lz4), hybrid strategies
- **GPU-native kernels** — PyTorch implementations for 7 compressors with zero-copy CUDA timing
- **Real KV caches** — benchmark with actual model activations from WikiText-103, GSM8K, or C4 datasets (not just synthetic random data)
- **End-to-end generation quality** — perplexity delta and token agreement metrics via KV cache injection, beyond cosine similarity
- **Multi-GPU calibration** — auto-detects GPU properties, supports cross-GPU comparison (T4 vs A100 vs H100)
- **Transfer model validation** — real TCP localhost measurements to validate the analytical transfer model
- **Serving framework integration** — reference vLLM adapter (`KVShuttleConnector`) for disaggregated prefill/decode
- **Analytical transfer simulation** — models sequential and pipelined transfer at configurable bandwidths
- **Break-even analysis** — identifies the maximum bandwidth at which each strategy is beneficial
- **Multi-model sweep** — benchmarks across 6 model architectures (Qwen2.5-3B through Llama-3.1-8B)
- **Learned router** — trains a lightweight classifier to select the best compressor per-request

## Project Structure

```
kvshuttle/
├── compression/       # 14+ compressors (BaseCompressor ABC, decorator registry)
│   ├── registry.py    # @register("name") decorator pattern
│   ├── uniform_quant.py, fp8.py, kivi.py, cachegen.py, ...
│   └── torch_quant.py # GPU-native PyTorch kernels
├── datasets.py        # Real prompt loading (wikitext, gsm8k, c4)
├── evaluation/        # Quality metrics (cosine sim, perplexity, token agreement)
├── models/            # KV cache extraction + injection for forward pass eval
├── profiling/         # Timing and memory measurement
├── router/            # Learned + oracle compressor selection
├── serving/           # vLLM adapter (KVShuttleConnector)
├── transfer/          # Transfer simulation, pipelining, and real TCP validation
└── visualization/     # Paper figure generation (quality, GPU calibration, transfer)

experiments/
├── configs/           # YAML experiment configurations
├── notebooks/         # GPU calibration Colab notebook + results
├── results/           # Benchmark outputs (model_sweep, real_kv_sweep, etc.)
└── scripts/           # Run experiments, generate paper assets, demos

paper/
├── figures/           # Generated PDFs (gpu_calibrated/, model_sweep/, router/)
└── tables/            # Generated LaTeX tables

tests/                 # pytest suite (79 tests)
```

## Installation

```bash
pip install -e .

# For development
pip install -e ".[dev]"
```

Requires Python >= 3.11. GPU kernels require PyTorch with CUDA support.

## Quick Start

### Run a smoke test

```bash
python experiments/scripts/run_experiment.py experiments/configs/smoke_test.yaml
```

### Run the full model sweep

```bash
python experiments/scripts/run_experiment.py experiments/configs/model_sweep.yaml
```

### Run with real KV caches

```bash
python experiments/scripts/run_experiment.py experiments/configs/real_kv_sweep.yaml
```

This uses WikiText-103 prompts instead of synthetic data, and evaluates perplexity delta and token agreement alongside cosine similarity.

### GPU calibration

Open `experiments/notebooks/gpu_calibration.ipynb` in Google Colab, select your GPU runtime (T4, A100, or H100), and run all cells. The notebook auto-detects GPU tier and adapts configuration (sequence lengths, model configs, bandwidth test range, repeat count). Results are saved as `gpu_calibration_results_{gpu_tier}.json` (e.g., `gpu_calibration_results_a100.json`).

### Generate paper assets

```bash
# Single GPU (T4 baseline)
python experiments/scripts/integrate_gpu_calibration.py \
    experiments/results/model_sweep/results.json \
    experiments/notebooks/gpu_calibration_results_t4.json

# Multi-GPU comparison (T4 + A100)
python experiments/scripts/integrate_gpu_calibration.py \
    experiments/results/model_sweep/results.json \
    experiments/notebooks/gpu_calibration_results_t4.json \
    --gpu-results experiments/notebooks/gpu_calibration_results_a100.json
```

### Validate transfer model

```bash
python experiments/scripts/validate_transfer_model.py
```

Compares the analytical `time = size / bandwidth` model against real TCP localhost measurements across payload sizes from 1 KB to 100 MB. Generates `paper/figures/transfer_validation.pdf`.

### vLLM integration demo

```bash
python experiments/scripts/demo_vllm_integration.py [model_name] [compressor_name]
```

End-to-end demo: prefill on a real model, compress KV cache, simulate transfer at various bandwidths, decompress, and evaluate quality. Works standalone without vLLM installed.

## Compressor Registry

Compressors are registered via decorator and looked up by name:

```python
from kvshuttle.compression.registry import get_compressor, list_compressors

print(list_compressors())
comp = get_compressor("uniform_int8")
compressed = comp.compress(keys, values)
keys_out, values_out = comp.decompress(compressed)
```

## Serving Integration

The `KVShuttleConnector` provides a simple send/receive interface for integrating KV cache compression into disaggregated serving frameworks:

```python
from kvshuttle.serving.vllm_adapter import KVShuttleConnector

connector = KVShuttleConnector("kivi_2bit")

# Prefill node: compress and serialize
wire_bytes = connector.send_kv_cache(keys, values)

# Transfer wire_bytes over network...

# Decode node: deserialize and decompress
keys_out, values_out = connector.recv_kv_cache(wire_bytes)
```

## License

MIT
