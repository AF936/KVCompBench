# BF16 benchmark / BF16 基准测试

发布范围为 Linux x86-64 BF16 执行器。FP8 研究文章和本执行器的验证范围不同。本仓库不包含原始 KV 数据、虚拟环境、二进制依赖和 JSONL 实验记录。

Linux setup: `bash setup_paper_codecs.sh` from the repository root. Python 3.10-3.12, GCC, and a CUDA-13-compatible driver are required. The setup downloads dependencies and verifies pinned upstream sources. Windows packaging and FP8 execution are outside this release's validated scope.


## 配置 / Configuration

`settings.txt` 的 `[benchmark_datasets]` 将 `input/` 中的连续 BF16 Tensor 文件名映射到每块字节数。当前 DeepSeek 为 131072 字节（128 KiB），Qwen 为 524288 字节（512 KiB）。新增数据集只需添加文件与对应配置。

`[benchmark].cpu_encode_threads` 控制 CPU 编码工作线程数；1 表示单工作线程。它不等于原生 codec 内部线程数。`decode_batch_blocks` 控制 GPU 解码每批独立块数，当前为 256；CPU 解码逐块计时。尾块和尾批允许小于配置值。仅测试支持 BF16 的 codec，不隐式转换为其他 dtype。

Dataset entries map contiguous BF16 tensors under `input/` to block sizes in bytes. Current sizes are 128 KiB for DeepSeek and 512 KiB for Qwen. CPU encoding workers and GPU decode batch size are independent. Native codec threading may differ from framework workers. Only BF16-capable codecs are tested.

## 启动与续跑 / Execution and resume

项目目录以下用 `/path/to/KVCompBench` 表示。须先获得 GPU 分配或本次空闲 GPU 使用授权。下例中的 GPU UUID 必须替换为已分配设备；输出目录必须位于项目的 `benchmark/output/` 内。

```bash
cd /path/to/KVCompBench && \
CUDA_VISIBLE_DEVICES='GPU-REPLACE-WITH-ASSIGNED-UUID' \
PYTHONDONTWRITEBYTECODE=1 \
.venvs/linux-x86_64-paper-cuda13/bin/python \
benchmark/run_full_cpu_encode_gpu_decode.py \
--output benchmark/output/full_run_name
```

`--output` 为必填参数。程序直接读取 `settings.txt`；环境名称仅供启动器选择解释器，不能改变已经启动的 Python 环境。中断后以相同输出目录重新执行命令即可续跑。新一轮完全独立的测试使用新的空目录。`--validate-only` 只运行小样本验证，不代表完成全量。

The output argument is required. Run with the project interpreter and an assigned GPU. Reuse the same output directory to resume, or select a new empty directory for an independent complete run. Validation mode uses small samples only. No Docker operation is required.

## 编解码路径 / Encoding and decoding paths

路径前项表示编码设备，后项表示解码设备：CPU-CPU、CPU-GPU、GPU-GPU、GPU-CPU。基础路径由插件及后端声明决定。兼容的另一 CPU 解码入口通过插件目录中的 `decode_backends.json` 声明，执行器按统一规则增加测试项目，不根据 codec 名称添加补测分支。

CPU 解码必须接受同一编码路径产生的原始压缩帧；不能通过更换压缩算法或编码封装代替另一解码端。每条路径独立实测，并逐块进行原始字节一致性校验。没有声明兼容 CPU 解码器的 GPU 编码格式不自动生成 GPU-CPU 测量值。

当前完成的全量包含 40 个编解码配置：20 个 CPU-CPU、10 个 CPU-GPU、10 个 GPU-GPU；两份数据共 80 项汇总。此计数描述现有配置，不是框架的固定限制。GPU-CPU 的可用性以具体帧格式及 CPU 解码声明为准。

The first device encodes and the second decodes. Additional CPU decoding is declared per plugin and tested against the existing encoded frame format. Each path is measured and byte-verified independently. Current results cover 40 configurations across two datasets. Unsupported paths are not replaced with a different algorithm.

## JSONL 复用与保留 / JSONL reuse and retention

`result.csv` 仅为派生汇总输出，绝不用于判断是否跳过测试。唯一的块级复用依据是 `progress.jsonl` 中匹配项目指纹的已验证记录。

项目指纹记录数据文件内容哈希、codec 目录内执行器实际扫描的源码与声明文件、声明的依赖版本、编解码路径、块大小、CPU 工作线程数、解码批大小、计时版本和运行环境。具体内容可查阅对应的 `job_*.json`。当前扫描的插件文件类型为 `.py`、`.json`、`.so`、`.dll`、`.pyd`，不扫描 `__pycache__`；未声明的外部依赖不保证能够自动检测变化。

- 新增独立 codec 不改变其他 codec 的项目指纹，已有匹配记录继续复用。
- 修改 codec 扫描范围内的实现或声明，使该 codec 相关项目重新计算指纹。
- 数据或公共执行器、计时实现、环境变化可能影响多个项目。
- 不匹配的旧记录保留在 JSONL 中，不用于当前汇总；新记录追加写入。
- 一个 GPU 解码批次按一行 JSONL 提交，只有通过逐字节校验的记录才提交。

完成后永久保留 `progress.jsonl`、`job_*.json`、`capabilities.json` 和结果文件，不自动删除 JSONL。历史 CSV 若没有对应 JSONL，不能恢复当时的块级计时记录，也不会被当成已完成块使用。复用时不重新校验旧块的实际解码，依据的是当时已通过校验的匹配记录。

CSV is output only. Resume matches verified JSONL blocks by per-project fingerprints. Independent new plugins leave existing project identities unchanged. Data, shared implementation or environment changes may invalidate several projects. Old records remain preserved. The exact fingerprint specification is stored in `job_*.json`; undeclared external changes are not guaranteed to be detected. Completed journals are retained permanently.

## 计时与公平性 / Timing and comparability

计时版本为 `codec_call_v2`。吞吐量＝原始未压缩字节数 ÷ 计时秒数 ÷ 1048576，单位 MiB/s。编码和解码使用相同的原始字节数作为分子。

CPU 使用 `perf_counter` 测量输入准备好后的 codec 调用边界；排除预热及外部准备。多工作线程聚合编码吞吐按同组调用的时间窗口计算。GPU 输入先驻留显存，在实际执行流上使用 CUDA Event 测量调用至计算完成；排除预热、H2D/D2H、外部封装和结果校验。API 内必要分配和处理计入调用窗口。批与批之间目前串行，不包含跨批流水线重叠。

比较结果须核对数据、块大小、路径、线程、批大小、版本与运行环境。CPU、GPU 解码路径均使用实际测量值，不以另一条路径的吞吐量替代。

Throughput uses original uncompressed bytes and the codec-call timing window. CPU uses host timers; GPU uses CUDA events with device-resident input. Warmup, external transfers, framing and verification are excluded. Compare measurements only under matching configurations and environments.

## 结果与 HTML / Results and HTML

每个输出目录保存 `result.csv`、`heatmap.png`、`heatmap.html`、`progress.jsonl`、`job_*.json` 和 `capabilities.json`。CSV 在项目完成时更新；PNG 和 HTML 在整轮成功完成后生成，随后更新 `benchmark/output/latest.html`。日志及启动记录由启动方式决定。

HTML 支持 CPU/GPU 解码端、指标和执行路径选择。没有对应方案或实测结果的单元格显示“/”。PNG 是压缩比静态总览；编码及解码吞吐量见 HTML 或 CSV。codec 独立备注可放在 `benchmark_notes.json`，由报告读取。离线 HTML 内嵌数据和图片；服务器更新后，本机副本需要重新复制。

最新已完成结果：`benchmark/output/generic_full_20261115_114333/`。旧结果目录保留。重建已有报告不需要 GPU：

```bash
cd /path/to/KVCompBench && \
.venvs/linux-x86_64-paper-cuda13/bin/python benchmark/html_report.py \
benchmark/output/generic_full_20261115_114333 \
--latest benchmark/output/latest.html
```

CSV is updated after each completed project. PNG and offline HTML are generated after the entire run succeeds. HTML provides decoder, metric and route selection; missing measurements show a slash. The stable latest report is `benchmark/output/latest.html`.

## 新增 codec / Adding codecs

在 `compressors/` 下建立独立目录，提供 `codec.json` 和 API 实现，声明名称、接口、dtype 及版本。优先使用现有共享后端；自定义适配放在插件目录，必须满足对应帧格式。按需提供 `runtime_requirements.json`、`decode_backends.json` 和 `benchmark_notes.json`。

CPU 解码声明可使用共享格式处理器，或插件内的 module 入口。插件维护者负责声明真实格式兼容性和依赖；执行器对所有声明的路径使用同样的计时、校验和 JSONL 匹配流程。新增 codec 应先通过小样本，再执行全量。没有对应的解码实现时保留缺失状态，不改换算法生成结果。

Create a codec-local manifest and API, declare supported dtypes and versions, and reuse compatible shared backends where possible. Declare additional CPU decoding and dependencies explicitly. All registered paths use the same timing, lossless checks and journal matching. Validate small samples before full execution.
