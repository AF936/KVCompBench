# Framework modules

The public harness targets Linux x86-64 BF16 execution. See [the benchmark guide](../ReadMe.md) for installation, timing, decoder declarations and JSONL resume.

`core.py`, `plugins.py` and `adapters/` provide codec discovery and byte APIs. `gpu_decode.py`, `gpu_codec.py`, `batch_decode.py` and `codec_timing.py` implement frame dispatch and measured execution. `cpu_decode_registry.py` resolves plugin-local CPU decoder declarations.
