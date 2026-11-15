# KVCompBench

**KV Cache Compression Benchmark**

KVCompBench is an open, extensible benchmark for KV cache compression. **We are starting with lossless compression**, building reproducible benchmarks for BF16, FP8 (E4M3FN/E5M2), and different KV layouts before expanding to other compression approaches. The evaluation workflow is intended to be adaptable to **LMCache and other KV storage/serving backends**.

**Current development track:** [KVCompBench-lossless](https://github.com/hpdps-group/KVCompBench/tree/KVCompBench-lossless). This first stage focuses on establishing lossless codec support, correctness checks, consistent compression-ratio accounting, and CPU/GPU performance measurements.

**Website:** https://hpdps-group.github.io/KVCompBench/

**First technical study:** [Lossless Compression of FP8 KV Caches: What Actually Helps?](https://hpdps-group.github.io/KVCompBench/blog/fp8-lossless-2026/)

## Lossless benchmark: first priority

A compression ratio alone does not tell a serving system which lossless codec to use. Codec choice depends on payload format, chunk/frame geometry, encoding overhead, decoder availability, and the effective I/O path. We are building a framework to compare these factors using the same data and accounting conventions.

We aim to measure:

- **Lossless correctness:** exact restoration of raw KV bytes.
- **Compression ratio:** counting required per-frame overhead, with transparent treatment of scales and descriptors.
- **CPU encoding throughput:** measured in uncompressed KV GB/s, with worker count and setup scope explicit.
- **GPU decoding throughput:** measured in uncompressed KV GB/s, distinguishing GPU-resident codec timing from host-to-device transport.
- **Cache geometry:** attention layout, independently decodable chunk sizes, and layer/group variation.
- **Applicability:** which codecs work for BF16, E4M3FN, E5M2, and model-specific KV layouts.

## First release: FP8 benchmark research note

The first lossless study uses FP8 KV payload exports from Qwen and DeepSeek and compares Zstd levels, Deflate, LZ4, LZ4-HC, Snappy, and CPU-only Brotli. It also examines page-size sensitivity, layer-specific compressibility, and reversible preprocessing. The article explicitly states which benchmarks include GPU decompression and which do not.

- [Read the study](https://hpdps-group.github.io/KVCompBench/blog/fp8-lossless-2026/)
- [Read the Markdown source](docs/blog/fp8-lossless-2026/article.md)
- [Discuss lossless compression in LMCache RFC #4498](https://github.com/LMCache/LMCache/issues/4498)

## Development roadmap

| Area | Status |
| --- | --- |
| FP8 technical study and benchmark methodology | **Published** |
| Linux BF16 lossless benchmark harness | Source released; FP8 harness integration remains planned |
| Machine-readable BF16 results | RTX 5090 reference run released |
| Independent prompts / trace-based validation | Planned |
| LMCache and other backend adapters | Planned |
| End-to-end L2 retrieval measurements | Planned |
| Other KV compression approaches | Future scope; not yet implemented |

**Scope note:** The current focus is **lossless compression**. This first release includes the technical article and figures. The Linux BF16 harness and a machine-readable reference result are now available below. Raw KV captures are not distributed; the FP8 article is a separate study and does not imply FP8 execution support in this harness.

## Collaboration

We welcome feedback on codec baselines, BF16/FP8 KV layouts, lossless GPU decoder interfaces, representative public data, and fair benchmarking methodology. You can [open an issue](https://github.com/hpdps-group/KVCompBench/issues) or discuss interface alignment in [LMCache RFC #4498](https://github.com/LMCache/LMCache/issues/4498).

This is an independent research and benchmarking effort, **not an official LMCache project**. We hope it will be useful to LMCache and other KV cache systems.

## Article engagement counters

The published FP8 article includes a small page-view counter and a like button at the bottom. Both use the public, no-account [Count API](https://countapi.mileshilliard.com/) service.

- **Views:** One request per browser tab session (reloading the same tab does not increment again).
- **Likes:** One like per browser profile, tracked locally using browser storage.
- **Data limits:** Counters are public, unauthenticated and approximate; they are neither unique-visitor analytics nor protected against deliberate manipulation.
- **Privacy:** Browsers call the counter provider over HTTPS; this site does not collect visitor identities or set tracking cookies.

The counters degrade quietly if the external service is unavailable.

## Linux BF16 benchmark harness

The [benchmark guide](benchmark/ReadMe.md) documents configuration, CPU/GPU paths, timing and JSONL resume. The [RTX 5090 reference result](examples/results/rtx5090_bf16_20261115/heatmap.html) contains 80 dataset/path measurements with CPU/GPU decoder selection. [CSV](examples/results/rtx5090_bf16_20261115/result.csv) and [run metadata](examples/results/rtx5090_bf16_20261115/metadata.json) accompany the report. Download the HTML file to open it offline.

```bash
# Linux x86-64; Python 3.10-3.12, a C compiler, and a CUDA-13-compatible driver.
bash setup_paper_codecs.sh
# Place contiguous BF16 .pt tensors in benchmark/input/ and edit settings.txt.
CUDA_VISIBLE_DEVICES=0 .venvs/linux-x86_64-paper-cuda13/bin/python \
  benchmark/run_full_cpu_encode_gpu_decode.py --output benchmark/output/my_run
```

Only use an assigned GPU on shared systems. The runner reads `benchmark/settings.txt`; an example configuration is included. The output directory must stay under `benchmark/output/`.

`result.csv` is output only. `progress.jsonl` retains verified block transactions, and each `job_*.json` describes the input, codec and execution fingerprint. Independent plugins do not invalidate each other's matching records. Missing decoder implementations are displayed as `/`, never substituted with another algorithm.

The published run covers two BF16 inputs and 40 encoding/decoding configurations. Raw KV tensors, virtual environments, compiled binaries and block journals are excluded. The reference result cannot reproduce the original private inputs by itself. Linux is the supported release target; Windows and FP8 are not claimed as validated harness targets. Packaging checks do not constitute a clean-machine GPU rerun.

Third-party vendored sources retain their upstream licenses. Dependency versions and source checksums are recorded in manifests; installers download dependencies rather than committing environment files.
