# Third-party sources and dependencies

Vendored upstream files retain their original license notices. Codec adapters and benchmark orchestration are separate from upstream compression implementations.

| Component | Source reference | Distribution |
|---|---|---|
| fse | Cyan4973/FiniteStateEntropy 9f30e0918f87bd835fa040d922a208d7b219e50b | Vendored source subset with upstream LICENSE and source_hashes.json |
| streamvbyte | fast-pack/streamvbyte 60286c56cc61f16dfbe003d6e1ac84ad48ad4e8b | Vendored source subset with upstream LICENSE and source_hashes.json |
| splitzip | Intelligent-Microsystems-Lab/SplitZip 42c743118a29db3270ae6a921a4a4eecb20cad04 | Vendored source subset with upstream LICENSE and source_hashes.json |
| zipnn | https://files.pythonhosted.org/packages/97/3c/08fd51416b56a40e3da4cb968b2b1ad424a7123db5c82397076f4d199390/zipnn-0.5.4.tar.gz 0.5.4 | Downloaded and SHA-256 checked during installation |
| OpenZL | https://github.com/facebook/openzl — package version 0.3.0 | Installed dependency; no binary included |

Other Python dependencies, PyTorch, Triton and NVIDIA nvCOMP are installed separately and retain their respective licenses. Their installation and use are subject to upstream terms.

The BF16 publication omits FP32/FP64-only experimental plugins and machine-specific installation artifacts. The bundled 5090 measurements precede publication packaging changes; they are historical measurements, not a rerun of the packaged tree.

ZipNN 0.5.4 is built with a CPython method-table terminator fix and identified as 0.5.4+sentinel1. The FSE/Huff0 adapter uses subframes for outer blocks larger than Huff0's supported chunk size. Frame overhead is counted in compressed size.

The publication also repairs truncated final JSONL transactions by retaining the incomplete fragment in a .jsonpart file and preserving complete transactions. This recovery behavior is covered by unit tests.
