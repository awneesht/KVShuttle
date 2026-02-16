# KVShuttle

Benchmark and decision framework for KV cache transfer compression in disaggregated LLM serving.

KVShuttle evaluates 14+ compression strategies across multiple models and sequence lengths, providing GPU-calibrated timing data and analytical transfer simulation to help practitioners choose the right compression scheme for their bandwidth regime.

## Key Results

| Strategy | Ratio | Key cos | Val cos | GPU Comp. | GPU Decomp. | Speedup |
|---|---|---|---|---|---|---|
| uniform_int8 | 2.0x | 0.9998 | 0.9998 | 0.33 ms | 0.07 ms | 34-80x |
| kivi_2bit | 6.5x | 0.9649 | 0.8706 | 0.42 ms | 0.17 ms | 53-68x |
| uniform_int4 | 3.6x | 0.9872 | 0.9934 | 5.28 ms | 1.29 ms | 62-63x |
| fp8_e4m3 | — | — | — | — | — | 36-64x |
| cachegen | 3.5x | 0.9927 | 0.9854 | 3.36 ms | 0.76 ms | 33-56x |
| cascade_prune50_int4 | 7.1x | 0.7370 | 0.7021 | 2.81 ms | 0.85 ms | 48-60x |
| palu_lr | 2.4x | 0.9829 | 0.9778 | 185 ms | 0.03 ms | 5-216x |

*GPU timings measured on Tesla T4 with CUDA Event timing (zero-copy, no CPU round-trips).*

## Features

- **14+ compressors** — identity, uniform INT8/INT4, FP8, KIVI 2-bit, CacheGen, KVQuant, Palu (SVD), token pruning, cascade (prune+quantize), lossless (zstd/lz4), hybrid strategies
- **GPU-native kernels** — PyTorch implementations for 7 compressors with zero-copy CUDA timing
- **Analytical transfer simulation** — models sequential and pipelined transfer at configurable bandwidths
- **Break-even analysis** — identifies the maximum bandwidth at which each strategy is beneficial
- **Multi-model sweep** — benchmarks across 6 model architectures (GPT-2 through Llama-3.1-8B)
- **Learned router** — trains a lightweight classifier to select the best compressor per-request

## Project Structure

```
kvshuttle/
├── compression/       # 14+ compressors (BaseCompressor ABC, decorator registry)
│   ├── registry.py    # @register("name") decorator pattern
│   ├── uniform_quant.py, fp8.py, kivi.py, cachegen.py, ...
│   └── torch_quant.py # GPU-native PyTorch kernels
├── evaluation/        # Quality metrics (cosine sim, attention error, perplexity)
├── models/            # KV cache extraction from HuggingFace models
├── profiling/         # Timing and memory measurement
├── router/            # Learned + oracle compressor selection
├── transfer/          # Transfer simulation and pipelining
└── visualization/     # Paper figure generation

experiments/
├── configs/           # YAML experiment configurations
├── notebooks/         # GPU calibration Colab notebook + results
├── results/           # Benchmark outputs (model_sweep, compression_sweep, etc.)
└── scripts/           # Run experiments, generate paper assets

paper/
├── figures/           # Generated PDFs (gpu_calibrated/, model_sweep/, router/)
└── tables/            # Generated LaTeX tables

tests/                 # pytest suite
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

### GPU calibration

Open `experiments/notebooks/gpu_calibration.ipynb` in Google Colab (T4 runtime), run all cells, and download `gpu_calibration_results.json`.

### Generate paper assets

```bash
python experiments/scripts/integrate_gpu_calibration.py
```

This reads `experiments/notebooks/gpu_calibration_results.json` and `experiments/results/model_sweep/results.json` to produce all figures and tables in `paper/`.

## Compressor Registry

Compressors are registered via decorator and looked up by name:

```python
from kvshuttle.compression.registry import get_compressor, list_compressors

print(list_compressors())
comp = get_compressor("uniform_int8")
compressed = comp.compress(kv_cache)
restored = comp.decompress(compressed)
```

## License

MIT
