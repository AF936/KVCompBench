# KVCompBench

[Repository](https://github.com/AF936/benchmark_basic) · [Upstream](https://github.com/hpdss-group/KVCompBench)

## 范围 / Scope

面向 KV Cache 数据的无损压缩基准。采用显式 KV 页面或通用字节分块，校验每个解码结果，并比较压缩比、编码及解码吞吐量。输入数据、实验结果、虚拟环境和机器配置不随仓库发布。

A lossless KV-cache compression benchmark with explicit KV pages, a flat-block baseline, byte-exact verification, resumable journals and offline reports. Datasets, results, environments and machine-specific settings are excluded from version control. No precomputed results are bundled.

## 目录 / Layout

```text
ReadMe.md
run_benchmark.sh              # full benchmark entrypoint
run.sh                        # generic compression CLI
setup_basic.sh                # minimal Linux CPU installation
setup_paper_codecs.sh         # optional Linux CUDA/native installation
benchmark/
  settings.example.txt        # tracked configuration example
  input/                      # place datasets here
  output/                     # generated experiment directories
  compressors/<package>/      # codecs and their local backends
    codec.json
    package.json              # core/optional classification
    benchmark.json            # optional route declaration
  compression_framework/      # scheduling, timing and verification
  requirements/               # separate dependency groups
  templates/                  # report.html, report.css, report.js
  tests/test_repository_layout.py
```

## 基础安装 / Minimal installation

参考平台为 Linux x86-64、64 位 Python 3.10+。从仓库根目录运行，不需要 GPU，也不安装 nvCOMP、Triton、OpenZL、ZipNN 或 Java 编解码器。

The reference minimal profile is Linux x86-64 with 64-bit Python 3.10+. It uses CPU PyTorch, NumPy and standard-library codecs; CUDA and optional native codecs are not required.

```bash
cp benchmark/settings.example.txt benchmark/settings.txt
bash setup_basic.sh
```

只提交 `benchmark/settings.example.txt`。`settings.txt` 是本机配置，已被 `.gitignore` 排除。不要把访问令牌、私有路径、数据或环境目录提交到 Git。

Only the example is versioned. Create and edit your own ignored `settings.txt`; do not commit credentials, private paths, datasets or virtual environments.

## 核心与扩展 / Core and optional packages

`compressors/<package>/package.json` 声明 `tier: core` 或 `tier: optional`。核心集为标准库基线；第三方及原生实现属于可选扩展。初始清单保存在 `benchmark/codec_sets.json`；运行时以各插件的声明为准。未分类的新插件默认属于可选扩展，无须修改框架中的 codec 名单。

Each package declares its tier in `package.json`. The core set contains standard-library baselines; third-party, GPU and native codecs are optional. `benchmark/codec_sets.json` is an initial inventory; runtime discovery reads the package declarations. A new unclassified plugin defaults to optional; the runner has no hard-coded codec allowlist.

配置 `codec_set = core` 只选择核心包；`codec_set = all` 选择所有支持 BF16 的包。`execution_device = cpu` 使用各插件自身的 CPU API 和帧格式，并排除只有 GPU 编码的插件；`execution_device = declared` 使用插件声明的路径及兼容 CPU 解码器。两者是不同的测试配置，指纹和帧格式会区分它们，不把不兼容格式视为同一配置。

`codec_set` controls package selection. `execution_device = cpu` selects the plugin CPU API and its frame format; GPU-only encoders are excluded. `declared` selects codec-local benchmark routes and compatible CPU decoders. These are distinct configurations with distinct fingerprints and declared frame formats.

依赖文件 / Dependency groups:

| File | Purpose |
| --- | --- |
| `benchmark/requirements/core.txt` | NumPy and lightweight framework utilities |
| `benchmark/requirements/optional-cpu.txt` | Third-party CPU codec packages |
| `benchmark/requirements/optional-gpu-cu13.txt` | nvCOMP CUDA 13 |
| `benchmark/compression_framework/requirements-paper.txt` | Explicit aggregate for the optional full profile |

PyTorch 的 CPU/CUDA 发行包分别从对应官方索引安装。原生扩展还需要相应的编译器或 JDK；只在选择扩展测试时安装。`setup_paper_codecs.sh` 是 Linux CUDA 13 的完整扩展安装入口，包含第三方/native 依赖，不是基础安装步骤。

Install the matching CPU/CUDA PyTorch distribution from its official index. Native extensions may require compilers or a JDK. The full Linux CUDA 13 installer is optional:

```bash
CUDA_VISIBLE_DEVICES='GPU-<assigned-uuid>' bash setup_paper_codecs.sh
```

扩展安装器会选择 `linux-x86_64-paper-cuda13` 环境。安装后，在本机 `settings.txt` 中将 `codec_set` 改为 `all`，将 `execution_device` 改为 `declared`。确保运行环境实际包含选择的扩展依赖。依赖缺失默认报错；只有显式设置 `skip_unavailable_codecs = true` 时才跳过并记录原因。

The full installer selects the `linux-x86_64-paper-cuda13` environment. Set `codec_set = all` and `execution_device = declared` in local settings. Missing dependencies fail by default; skipping requires explicit `skip_unavailable_codecs = true`, with reasons recorded in the output.

## 数据和页面 / Data and pages

将 BF16 `.pt` 数据放在 `benchmark/input/`，按真实张量轴配置 `[dataset:<filename>]`。示例中的文件名是待提供的输入，不代表仓库包含这些数据。

Place BF16 `.pt` tensors in `benchmark/input/` and declare their actual axes in dataset sections. Example filenames describe user-supplied inputs; those files are not bundled.

`layout = kv` 保留 layer、batch、head、K/V 和 token 边界。示例 Qwen 形状 `[16,2,1,4,8192,256]`，完整 K/V 的 256-token 页为 1 MiB；DeepSeek 共享 latent 的 128-token 页为 128 KiB。末页按真实字节数计算。`layout = flat` 才使用 `[benchmark_datasets]` 的字节块大小。对由量化数据转换的 BF16 输入，应保留来源元数据，不能称为原始量化前 BF16。

KV pages retain logical boundaries and use the actual tail size. Byte block sizes apply only to `layout = flat`. Preserve provenance metadata for BF16 tensors derived from quantized data; they are not original pre-quantization BF16 tensors.

## 运行 / Run

```bash
# Small compatibility check; generated output is not a full performance result.
bash run_benchmark.sh --validate-only --output benchmark/output/validation

# Full core CPU run, using local settings.
bash run_benchmark.sh --output benchmark/output/core_full

# Optional declared GPU routes: first select all/declared and the CUDA environment.
CUDA_VISIBLE_DEVICES='GPU-<assigned-uuid>' bash run_benchmark.sh --output benchmark/output/full_run
```

GPU 命令中的占位 UUID 必须替换为已分配设备。没有 GPU 解码方案的 codec 保留 CPU 路径；不使用别的算法代替。报告用 CPU-CPU、CPU-GPU、GPU-GPU 或 GPU-CPU 标注路径，无对应实测数据时显示 `/`。

Replace the GPU UUID placeholder with an allocated device. Unsupported paths are not substituted with another algorithm. Reports label execution paths and display `/` when no corresponding measurement exists.

`cpu_encode_threads` 控制 CPU 编码并发数；有状态且不支持并发的后端使用串行保护。`decode_batch_blocks` 控制 GPU 解码批大小。设置、源码、帧格式、依赖或硬件变化会影响运行指纹。

## 计时和续跑 / Timing and resume

吞吐量为原始字节数除以计时时长，单位 MiB/s。CPU 使用 API 调用边界计时；GPU 使用实际流上的 CUDA Event。初始化、预校准、预热、传输和正确性校验排除在 codec 吞吐之外；页面吞吐单独包含准备、封装及传输。同一编码配置只测量一次编码，其帧交给兼容解码器，保留相同编码测量。

CPU timings measure API boundaries; GPU timings use events on the execution stream. Setup, calibration, warm-up, transfers and verification are outside codec throughput. Page throughput separately includes preparation and transfer overhead. Compatible decoders share the same encoder measurements and frame format.

中断后使用相同 `--output` 目录重启。唯一复用依据为 `progress.jsonl` 中指纹匹配且验证通过的记录，`result.csv` 不参与续跑判断。保留 JSONL、任务元数据及历史结果。临时帧仅在相关路径都验证完成后删除；恢复时可能重新生成帧并检查其哈希。

Resume by reusing the same output directory. Only fingerprint-matched, validated JSONL records authorize reuse. CSV is derived output. Keep journals and job metadata; temporary frames are removed after their dependent paths finish.

## 报告 / Reports

输出目录由运行创建。CSV 在配置完成后更新；整轮成功后生成 `heatmap.png`、`heatmap.html`，全量运行还更新 `benchmark/output/latest.html`。源文件在 `benchmark/templates/` 分离维护；发布 HTML 内嵌数据、CSS 和 JavaScript，可离线打开。

Output directories are created by runs. CSV is updated per completed configuration; final PNG/HTML reports are generated when the run succeeds. HTML source assets are separate, while generated reports remain self-contained.

从已有 CSV 和 PNG 重建报告（不需要 GPU）/ Rebuild an existing report without a GPU:

```bash
python3 benchmark/html_report.py benchmark/output/full_run --latest benchmark/output/latest.html
```

此示例目录只有在相应实验已经完成后才存在。

## 结构检查 / Repository checks

以下检查不需要数据、PyTorch 或 GPU / These checks require no dataset, PyTorch or GPU:

```bash
python3 -B benchmark/tests/test_repository_layout.py
```

## 新增 codec / Adding a codec

1. 在 `benchmark/compressors/<package>/` 新建包，提供符合现有 schema 的 `codec.json` 和 CPU API 或命令适配器。
2. 在独立 `package.json` 中声明 `tier`。没有该文件时默认 optional，不修改 `codec.json` 的 schema。
3. 将第三方依赖写入插件自己的 `requirements.txt` 或 `runtime_requirements.json`。不要加入核心依赖文件，除非它确实成为核心组件。
4. 如有 GPU 路径，按现有插件示例提供 `benchmark.json`、本地后端和可选 `decode_backends.json`。声明设备、帧格式和批能力；通用 runner 不添加按 codec 名称判断的分支。
5. 算法专属说明写入该插件的 `benchmark_notes.json`，报告自动显示。先运行小样本及真实末页验证，再跑全量；新增路径不得静默复用不同帧格式的数据。

Declare package metadata and implementations locally. Keep optional dependencies local, preserve frame compatibility, expose device/batch capabilities, and add plugin-specific notes in the plugin folder. The shared runner only selects, schedules, times and verifies.

## 仓库检查 / Repository checks

```bash
python3 benchmark/tests/test_repository_layout.py
```

该检查不创建 GPU 上下文，也不读取实验输入。验证核心 CPU round trips、插件分组、示例配置及模板布局。
