# Codec-local GPU Backend Template

Copy this directory into `benchmark/compressors/<codec_name>/`, replace every
placeholder, and assign unique codec and backend IDs. The module must expose
`cpu_encode(raw)`, `gpu_decode(payload, expected_bytes)`, and optionally
`gpu_encode(raw)`. Keep the payload self-describing if decoding requires
algorithm parameters; the framework only provides the versioned KVG1 envelope.

Validate CPU encode to GPU decode, GPU encode to GPU decode, short final
blocks, malformed payloads, zeros, random bytes, and production data exactly.
