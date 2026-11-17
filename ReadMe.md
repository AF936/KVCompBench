# KVCompBench

面向 BF16 KV Cache 的无损压缩基准，比较压缩比、编码和解码吞吐量，并逐页校验解码结果。

仓库：[hpdss-group/KVCompBench](https://github.com/hpdss-group/KVCompBench)

## 安装与统一入口

要求 64 位 Python 3.10+。Linux 使用 `python3`，Windows 可使用 `python` 或 `py -3`。原生构建目前支持 Linux x86-64，需预装 C/C++ 编译器；Java Codec 的固定 JDK 与 JAR 由框架下载。

```bash
python3 kvbench.py setup --codecs identity,gzip,zlib,bz2,lzma
python3 kvbench.py setup --codecs zstd:3,lz4,fse --plan
python3 kvbench.py setup --codecs zstd:3,lz4,fse
python3 kvbench.py codecs --status
python3 kvbench.py doctor --codecs gzip,zstd:3,lz4,fse
```

每个插件的 `install.json` 声明固定的直接依赖、源码提交、下载摘要和编译命令。主仓库保留 Adapter、Manifest 和必要的桥接代码，不包含 ALP、Chimp/ELF、FSE、StreamVByte、SplitZip 的第三方实现或 ZipNN 源码包。选择的算法及其依赖由框架自动安装，不需要手动接入 Adapter。

框架优先使用 `.venvs/<平台>-managed-<profile>/`，下载、源码、编译产物、安装收据和日志都保存在该环境内。安装只处理选中的 Codec 包，不会安装所有扩展。基础框架的 CPU PyTorch、NumPy 等依赖也由统一入口安装；已解析的依赖版本、编译器版本及产物摘要写入安装收据。

`--plan` 仅检查计划，不下载或创建环境。`codecs --status` 检查实际版本、路径、声明的接口及安装完整性，不创建 GPU 上下文；没有配置环境时会检查当前解释器。`doctor` 额外验证 CPU 无损往返。GPU-only Codec 在未显式启用 GPU 检查时标记为 deferred，不会把未测路径标记为已验证。

GPU 路径使用显式的 CUDA13 环境，并要求已分配设备：

```bash
python3 kvbench.py setup --codecs gzip,zlib,nvcomp-lz4 --device cuda13
CUDA_VISIBLE_DEVICES='GPU-已分配的UUID' python3 kvbench.py doctor --codecs gzip,zlib,nvcomp-lz4 --gpu
```

`doctor --gpu` 检查真实 GPU 接口、批量解码和兼容 CPU 解码，不是性能测量。纯 CPU 安装不会安装 CUDA 依赖。详细 Manifest 格式、外部安装和诊断语义见 [INSTALLATION.md](benchmark/compression_framework/INSTALLATION.md)。

## 配置与运行

安装后在本机 `benchmark/settings.txt` 中选择运行名单。安装选择与测试选择分开，安装 FP32-only Codec 不会把它自动加入 BF16 测试。

```ini
[benchmark]
codecs = identity, gzip, zlib, bz2, lzma
codec_set = core
execution_device = cpu
```

非空 `codecs` 优先于 `codec_set`，可直接包含 optional 名称，不需要修改 `package.json`。留空 `codecs` 时才按 core/all 分组选择；core 组包含五类经典算法的九个参数名称。重复、未知或当前数据类型/设备不兼容的名称会报错。缺少安装或版本不匹配也会报错，并提示通过统一入口安装。

```bash
python3 kvbench.py run --validate-only --output benchmark/output/core_validation
python3 kvbench.py run --output benchmark/output/core_full
python3 kvbench.py cli list
```

`execution_device = declared` 使用插件声明的路径。gzip、zlib 支持共享编码帧的 CPU/GPU 解码；identity、bz2、lzma 保留 CPU 路径。运行前先完成相应安装与诊断，并设置已分配 GPU 的 `CUDA_VISIBLE_DEVICES`。不同帧格式仍是独立配置。

默认使用框架管理环境。已有 Python 安装可显式验证并选用：

```bash
python3 kvbench.py setup --codecs zstd:3 --python /path/to/environment/bin/python
```

该方式不安装或修改外部依赖，要求版本匹配。已有原生安装还需提供显式外部锁，记录固定版本、产物摘要和路径；格式见安装文档。所有本机配置和诊断记录默认排除在版本控制之外。


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

输入数据、虚拟环境和本机配置不随仓库发布；运行产物默认不发布，已确认的结果快照位于 results/。根目录的安装与运行统一使用 `kvbench.py`，不再提供独立平台包装脚本。

新增 codec 在自己的目录声明 API、设备、帧格式、批量能力和依赖；兼容 CPU 解码声明使用 `decode_backends.json`，共享帧提供者使用 `frame_provider.json`。公共 runner 负责选择、调度、计时和正确性校验。


## 已发布结果

[2026-11-17 经典 BF16 CPU/GPU 全量结果](results/2026-11-17-classic-bf16/README.md)：26 项路径配置，72,618 块全部通过逐字节验证。结果快照不包含输入数据或本机配置。

[2026-11-17 GPU BF16 全量复测结果](results/2026-11-17-gpu-bf16/README.md)：20 个 GPU 配置覆盖两个完整数据集，含 40 条 GPU 路径与 20 条兼容 CPU 解码结果；111,720 个 GPU 页面全部通过逐字节校验。提供 Codec 与页面吞吐两种口径，测量提交为 `6ed1268`。
