# nvcomp

Integration code: `benchmark/compression_framework/adapters/catalog.py`.

Select a registered variant in `benchmark/settings.txt`. The installer provides the required Python API dependencies in the selected project environment.

Unified interface:

`compress(raw: bytes, metadata: dict) -> bytes`

`decompress(packet: bytes, expected_bytes: int, metadata: dict) -> bytes`

Official NVIDIA API: https://docs.nvidia.com/cuda/nvcomp/samples/nvcomp.html

Requires a CUDA profile, compatible NVIDIA driver, and an assigned GPU.
