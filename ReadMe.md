# KVCompBench

面向 BF16 KV Cache 的无损压缩基准，比较压缩比、编码和解码吞吐量，并逐页校验解码结果。

仓库：[hpdss-group/KVCompBench](https://github.com/hpdss-group/KVCompBench)

## 统一入口

根目录只使用 `kvbench.py`，要求 64 位 Python 3.10+。Linux 使用 `python3`，Windows 可使用 `python` 或 `py -3`。

```bash
python3 kvbench.py --help
python3 kvbench.py codecs
python3 kvbench.py setup
python3 kvbench.py run --validate-only --output benchmark/output/core_validation
python3 kvbench.py run --output benchmark/output/core_full
python3 kvbench.py cli list
```

基础安装创建项目内 CPU 虚拟环境，仅安装 CPU PyTorch 和核心依赖，不安装可选编解码器。安装会写入 `benchmark/settings.txt`；已有环境可通过该配置的 `environment.python` 指定解释器绝对路径。

## 默认与可选 codec

默认仅运行五个经典 CPU 基线：identity、gzip、zlib、bz2、lzma。zlib 和 lzma 包另有参数变体；留空 `codecs` 并设置 `codec_set = core` 时选择核心组的全部九个名称。

LZ4、Zstd、Snappy、FSE/Huff0、OpenZL、ZipNN、SplitZip、nvCOMP、StreamVByte、delta-zstd 及其他扩展继续保存在 `benchmark/compressors/`，默认不选中。插件由自己的 `package.json` 声明 `core` 或 `optional`；没有分类声明的插件视为 optional。支持数据类型也参与筛选，选择 all 不代表所有插件都支持 BF16。

`benchmark/settings.example.txt` 为基础配置示例。默认运行读取被 Git 忽略的 `benchmark/settings.txt`，使用精确的 `codecs` 名单和 `execution_device = cpu`。首次运行会从示例创建配置。可选扩展配置不预置；需要时复制示例为 `benchmark/settings.optional.local.txt` 并按需编辑。

```ini
[benchmark]
codecs = identity, gzip, zlib, bz2, lzma
codec_set = core
execution_device = cpu
```

非空 `codecs` 优先于 `codec_set`，可直接包含 `lz4`、`zstd:3` 等 optional 名称，无需修改插件分类。用 `python3 kvbench.py codecs` 查询精确名称。重复、空名称或当前数据类型/设备不支持的名称会报错。命令行 `--codecs` 覆盖配置名单；依赖仍需安装，显式选择的算法不会因 `skip_unavailable_codecs` 而静默消失。

将 `execution_device` 改为 `declared` 可使用插件声明的 CPU/GPU 路径：基础名单中的 gzip、zlib 同时测量兼容 CPU 与 GPU 解码并共享编码；identity、bz2、lzma 保留 CPU 解码。不同帧格式仍为不同配置。GPU 解码需要相应 CUDA PyTorch、nvCOMP 依赖和已分配的 GPU；纯 CPU 安装不包含 GPU 依赖。

完整扩展安装目前面向 Linux x86-64/CUDA 13，按需执行：

```bash
python3 kvbench.py setup --optional
python3 kvbench.py run --settings benchmark/settings.optional.local.txt --output benchmark/output/optional_full
```

扩展运行前，填写所需 `codecs` 名单；如需整个扩展组，留空 `codecs` 并设置 `codec_set = all`。按需使用 `execution_device = declared`，并安装相应依赖。GPU 使用必须先获得分配，设置 `CUDA_VISIBLE_DEVICES` 为已分配的 UUID。缺失依赖默认报错；分组选择时仅显式启用 `skip_unavailable_codecs` 才跳过并记录原因。

## 数据与页面

数据位于 `benchmark/input/`，配置中的 dataset 节明确张量轴。KV 模式按 layer/batch 与 token 范围切片，保留完整 K/V、head 和特征维；末页使用真实长度。Qwen 的 256-token 页为 1 MiB，DeepSeek 共享 latent 的 128-token 页为 128 KiB。具体大小取决于实际张量形状。flat 模式保留通用字节块基线。

由量化数据转换得到的 BF16 应保留来源元数据，不能称为原始量化前 BF16。

## 测量与续跑

初始化、校准、预热、传输和校验排除在 codec 吞吐之外；页面吞吐单独计入准备、封装和传输。CPU 使用调用边界计时，GPU 使用执行流上的 CUDA Event。SplitZip 使用首个页面校准后冻结的码本，并合并批次解码；小页面结果不能直接与大块 GPU 驻留输入结果比较。

同一编码配置的测量和帧由兼容 CPU/GPU 解码路径共享；不同帧格式是不同配置。复用相同输出目录可续跑，依据仅为指纹匹配且验证通过的 `progress.jsonl` 记录，CSV 不用于判断续跑。数据、依赖、源码或硬件改变可能导致指纹变化。

## 输出与维护

输出目录包含 `result.csv`、`progress.jsonl`、任务元数据及完成后生成的 `heatmap.html`、`heatmap.png`。HTML 内嵌数据、CSS 和 JavaScript，可离线打开。模板源文件位于 `benchmark/templates/`。

```bash
python3 benchmark/html_report.py benchmark/output/core_full
```

结果、数据、虚拟环境和本机配置不随仓库发布。根目录的安装与运行统一使用 `kvbench.py`，不再提供独立平台包装脚本。

新增 codec 在自己的目录声明 API、设备、帧格式、批量能力和依赖；兼容 CPU 解码声明使用 `decode_backends.json`，共享帧提供者使用 `frame_provider.json`。公共 runner 负责选择、调度、计时和正确性校验。


## 已发布结果

[2026-11-17 经典 BF16 CPU/GPU 全量结果](results/2026-11-17-classic-bf16/README.md)：26 项路径配置，72,618 块全部通过逐字节验证。结果快照不包含输入数据或本机配置。
