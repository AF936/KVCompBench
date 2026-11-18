# RTX 5090 BF16 and scaled-FP8 KV-page results (2026-11-18)

One complete pass across DeepSeek and Qwen BF16 caches and their E4M3FN/E5M2 requantizations produced 176 configurations: 116 GPU decoder paths and 60 compatible CPU decoder paths. All 491,568 decoded pages matched the complete original page bytes, including FP32 scales. The pre-run validation covered 176 configurations and 528 page decodes; its resume check passed. Four FP8/SplitZip combinations were explicitly skipped because SplitZip supports BF16 only.

## Measurement scope

- Hardware: one NVIDIA GeForce RTX 5090; Python 3.10.12, PyTorch 2.9.1+cu130 and CUDA 13.0. Per-codec dependency versions, frame formats, source hashes and job fingerprints are retained in `job_specs.json`.
- Twenty GPU-capable configurations were selected. This is a complete-input run of that selection, rather than every optional CPU-only codec. Compatible CPU/GPU decoders consume the same encoded frames and share each encoding measurement.
- DeepSeek uses shared latent pages of 128 tokens: 128 KiB for BF16 and 64 KiB data + 512 B FP32 scales for FP8 (64.5 KiB total), with a shorter tail page in each layer. Qwen uses separate K/V pages of 256 tokens: 1 MiB for BF16 and 512 KiB data + 8 KiB FP32 scales for FP8 (520 KiB total).
- FP8 files were requantized from BF16; these measurements do not establish that a model deployment natively emitted the same FP8 cache. Input metadata and quantization error metrics are retained. Compression is lossless relative to stored FP8 data and scales, not relative to the BF16 values before quantization.
- Codec-only timing excludes transfers and setup/calibration. Page timing is also recorded. Batch size is 256 pages and CPU encoding uses one thread. Results are one complete pass, with no independent performance repetitions or confidence intervals.

## Measured implementation

The server ran the exact X-drive FP8 implementation, deployed from a separately hashed source archive, using an existing CUDA environment. Its framework and runner hashes were checked against all 176 downloaded job specifications. `measured_source_sha256.json` identifies that source snapshot; `job_specs.json` additionally records codec files and runtime dependencies, including locally installed native files.

The branch also contains the FP8 implementation ported onto the manifest-managed installer. These performance results belong to the X-drive implementation and do not claim that the branch's installer or installation-identity preflight was exercised by this server run. No third-party source trees or native dependency binaries are added with this result snapshot.

## Files and reproducibility

- `result.csv`: all 176 measured configurations; `gpu_results.csv`: the 116 GPU decoder paths.
- `heatmap.html` and `heatmap.png`: original generated reports.
- `input_manifest.json`: six input hashes, sizes, tensor shapes, layouts and source/quantization metadata. Model names are retained as supplied metadata; exact checkpoint revisions and weight quantization settings were not independently established.
- `job_specs.json`, `environment.json`, `settings.example.txt` and `measured_source_sha256.json`: measured configuration and provenance. The example retains portable environment naming instead of the server's absolute interpreter path.
- `capabilities.json` and `unavailable_codecs.json`: available paths and explicit unsupported combinations.
- `summary.json`: verification totals and the SHA-256 of the downloaded JSONL journal. Every decode record was checked for sequential page coverage, and its aggregate data/scale/raw/compressed byte counts were matched to the CSV.
- `sha256.json`: checksums of the published files. Raw tensor files, the 609 MB continuation journal, and frame caches are excluded from Git history. The downloaded journal remains available in the local result directory.

With matching input files in `benchmark/input`, an authorized GPU selected through `CUDA_VISIBLE_DEVICES`, and a compatible installed CUDA environment, use `python kvbench.py run --settings results/2026-11-18-gpu-bf16-fp8/settings.example.txt --output benchmark/output/bf16_fp8_reproduction`. The source and package hashes in the measured job specifications define the original environment; changed source/dependency identities create a different experiment.
