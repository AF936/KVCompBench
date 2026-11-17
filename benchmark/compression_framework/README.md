# Unified KV Compression Framework

## Installation and Environment Selection

A 64-bit Python 3.10+ installation with `venv` support is required. Installation is project-local, without system package installation or driver changes.

From the project root on Linux:

```bash
bash setup_linux.sh --profile cpu
```

On Windows, run `setup_windows.bat` or:

```powershell
.\setup_windows.ps1 --profile cpu
```

Each system, architecture, and dependency profile has a separate environment:

```text
.venvs/
├── linux-aarch64-cpu/
├── linux-x86_64-cpu/
├── linux-x86_64-cuda12/
├── linux-x86_64-cuda13/
└── windows-x86_64-cpu/
```

Only the requested environment is installed. CPU profiles install NumPy, Zstandard, LZ4, Snappy, and fpzip. gzip, zlib, bz2, and lzma use the Python standard library. GPU profiles add the official NVIDIA nvCOMP Python package, including its library dependency. No PyTorch, model training, CuPy, custom CUDA kernels, or project reference compression algorithms are required by these API adapters.

Explicit GPU profiles:

```bash
bash setup_linux.sh --profile cuda12
bash setup_linux.sh --profile cuda13
```

A custom package index can be passed with `--index-url URL`. Package caches, build scratch files, lock files, and environment metadata stay under the corresponding environment. An existing environment is reused without deleting it. Environments must be recreated on a different machine or architecture; copying their directories is not supported.

After a successful installation, the script updates only the selected environment name in `benchmark/settings.txt`:

```ini
[environment]
name = linux-x86_64-cuda13
```

The launcher uses this exact environment:

```bash
bash run.sh list
bash run.sh compress kv.raw kv.kvc --codec zstd:3 --block-bytes 65536
bash run.sh decompress kv.kvc restored.raw
```

Windows uses `run_windows.bat` with the same arguments. Input/output paths and codec settings may also be supplied in `benchmark/settings.txt`. Existing output files are never overwritten.

## Shared RTX 5090 Server

Keep the project in a confirmed owned directory under `/home/liuxinyang` or `/data7t/lxy`. Example installation from an existing project directory:

```bash
cd /data7t/lxy/Accelerator && bash setup_server_5090.sh
```

The server profile checks Linux x86_64, project path, GPU model, and driver information. It selects CUDA 13 packages for driver branch 580 or newer, otherwise CUDA 12 packages for a supported RTX 5090 driver. Driver-family screening is a preflight check, not proof of runtime compatibility. The installer checks CPU APIs only and does not create a GPU workload. It never installs drivers, changes system CUDA, runs Docker, or selects a GPU by apparent availability.

After administrator GPU allocation and explicit task authorization, select the assigned device before a GPU check. Example for a device assigned as GPU 0:

```bash
cd /data7t/lxy/Accelerator && CUDA_VISIBLE_DEVICES=0 bash run.sh check --device gpu
```

Do not use the example device number unless that device was actually assigned. GPU codec names include `nvcomp-lz4`, `nvcomp-zstd`, `nvcomp-ans`, `nvcomp-snappy`, `nvcomp-deflate`, `nvcomp-gdeflate`, `nvcomp-gzip`, `nvcomp-bitcomp`, and `nvcomp-cascaded`. Support and alignment restrictions depend on the installed nvCOMP release. No 5090 runtime validation is claimed until the GPU check runs on that server.

Official references:

- https://docs.nvidia.com/cuda/nvcomp/installation.html
- https://docs.nvidia.com/cuda/nvcomp/samples/nvcomp.html
- https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html

## Drop-in Codec Packages

Place a codec package with `codec.json` in `benchmark/compressors/`; the next CLI invocation discovers it automatically. Python API bindings and command-line programs are supported without editing the global registry. See [PLUGINS.md](PLUGINS.md) for the package contract, dependency installation, and copyable templates. Select its name in `settings.txt` or with `--codec`.

## Codec Scope and Layout

```text
benchmark/
├── settings.txt
├── compressors/<codec>/API.md
└── compression_framework/
    ├── adapters/catalog.py
    ├── registry.json
    ├── requirements.txt
    ├── setup_environment.py
    ├── run_environment.py
    ├── core.py
    ├── cli.py
    └── test_*.py
```

`compressors/` contains per-codec package manifests, interface documentation, and optional codec assets. Built-in integration lives in `compression_framework/adapters/`; external Python APIs and command tools are loaded through `compression_framework/plugins.py`. General framework code stays outside codec packages.

Active entries wrap standard libraries, installable codec APIs, and optional official nvCOMP GPU APIs. Paper-only methods, training-dependent algorithms, custom project reference codecs, and local native wrappers without portable installation are excluded from the active registry. Excluded methods are not shipped with this framework.

fpzip requires native FP32 or FP64 payloads and is checked for bit-exact restoration before writing. GPU nvCOMP formats are identified by their own codec names; nvCOMP ANS is not claimed to reproduce an unrelated FSE or paper format. CPU and GPU streams are not assumed interchangeable.

## Configuration and Validation

Compression configuration includes codec, block bytes, input/output paths, and JSON tensor metadata. Relative configured paths resolve against `settings.txt`; explicit CLI paths resolve against the working directory. Decompression reads codec versions, block sizes, and metadata from the container. SHA-256 and pre-storage roundtrip verification are mandatory.

Input is a raw exported payload. Passing a `.pt` file compresses complete file bytes without identifying KV tensors. Caller-supplied shape, layer, and K/V metadata is preserved without semantic validation. No implicit dtype conversion is performed. Blocks, including the final block, must satisfy backend alignment requirements.

The ZIP_STORED container supports independent random block reads using `Reader.read_block(index)`. Blocks that do not shrink are stored raw. Reported stored bytes include metadata and index overhead. Failure can leave incomplete output; retry with a different target name. This framework is not hardened for untrusted containers.

```bash
bash run.sh check --device cpu --metadata '{"dtype":"fp32"}'
.venvs/linux-aarch64-cpu/bin/python -m unittest discover -s benchmark/compression_framework -v
```

Use the interpreter path corresponding to the current environment. No full-data throughput or GPU performance claims are made by these checks.
