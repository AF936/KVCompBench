"""Discover codec-local GPU backends declared in codec.json."""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
from dataclasses import dataclass
from pathlib import Path
import sys

from gpu_decode import Backend, backend, compress_cpu_to_gpu_frame, decode_gpu_frame, pack_frame, register_backend
from plugins import contained, manifests


@dataclass
class GpuCodec:
    name: str
    folder: Path
    declaration: dict
    dtypes: tuple[str, ...] | None = None
    _module: object | None = None

    @property
    def backend_id(self):
        if self.declaration["type"] == "shared":
            return self.declaration["backend"]
        return self.declaration["id"]

    def supports(self, dtype: str) -> bool:
        return self.dtypes is None or dtype in self.dtypes

    def _load_module(self):
        if self._module is not None:
            return self._module
        path = contained(self.folder, self.declaration["file"])
        unique = "kv_gpu_codec_" + hashlib.sha256(str(path).encode()).hexdigest()[:16]
        spec = importlib.util.spec_from_file_location(unique, path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot load GPU backend: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[unique] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(unique, None)
            raise
        decoder = getattr(module, self.declaration["gpu_decode"], None)
        encoder = getattr(module, self.declaration["cpu_encode"], None)
        if not callable(decoder) or not callable(encoder):
            raise ValueError(f"{self.name}: GPU backend callables are missing")
        try:
            backend(self.backend_id)
        except ValueError:
            register_backend(Backend(self.backend_id, f"codec-local GPU backend for {self.name}", decoder=decoder, cpu_encoder=lambda raw, options: encoder(raw)))
        self._module = module
        return module

    def encode_cpu(self, raw: bytes, dtype: str | None = None) -> bytes:
        if dtype is not None and not self.supports(dtype):
            raise ValueError(f"{self.name}: unsupported dtype {dtype}")
        if self.declaration["type"] == "shared":
            return compress_cpu_to_gpu_frame(raw, self.backend_id, self.declaration.get("encoder_options"))
        module = self._load_module()
        encoder = getattr(module, self.declaration["cpu_encode"])
        if dtype is not None and "dtype" in inspect.signature(encoder).parameters:
            return pack_frame(self.backend_id, len(raw), bytes(encoder(raw, dtype=dtype)))
        return compress_cpu_to_gpu_frame(raw, self.backend_id)

    def encode_gpu(self, raw: bytes) -> bytes:
        if self.declaration["type"] != "module":
            raise NotImplementedError(f"{self.name}: shared backend has no declared GPU encoder")
        module = self._load_module()
        function = getattr(module, self.declaration.get("gpu_encode", ""), None)
        if not callable(function):
            raise NotImplementedError(f"{self.name}: no GPU encoder declared")
        return pack_frame(self.backend_id, len(raw), bytes(function(raw)))

    def decode_gpu(self, frame: bytes, expected_bytes: int) -> bytes:
        if self.declaration["type"] == "module":
            self._load_module()
        return decode_gpu_frame(frame, expected_bytes)


def _validate(name: str, folder: Path, value: dict) -> dict:
    if not isinstance(value, dict) or value.get("frame_version") != 1:
        raise ValueError(f"{name}: gpu_backend requires frame_version 1")
    kind = value.get("type")
    if kind == "shared":
        if set(value) - {"type", "backend", "frame_version", "encoder_options"} or not {"type", "backend", "frame_version"} <= set(value) or not isinstance(value.get("backend"), str) or not isinstance(value.get("encoder_options", {}), dict):
            raise ValueError(f"{name}: invalid shared gpu_backend declaration")
        backend(value["backend"])
    elif kind == "module":
        required = {"type", "id", "file", "cpu_encode", "gpu_decode", "frame_version"}
        if set(value) - (required | {"gpu_encode"}) or not required <= set(value):
            raise ValueError(f"{name}: invalid module gpu_backend declaration")
        if not all(isinstance(value[key], str) and value[key] for key in ("id", "file", "cpu_encode", "gpu_decode")):
            raise ValueError(f"{name}: invalid module gpu_backend fields")
        contained(folder, value["file"])
    else:
        raise ValueError(f"{name}: gpu_backend type must be shared or module")
    return dict(value)


def discover_gpu_codecs(directory):
    found = {}
    for folder, spec in manifests(directory):
        declaration = spec.get("gpu_backend")
        if declaration is None or not spec.get("enabled", True):
            continue
        name = spec.get("name")
        if not isinstance(name, str) or name in found:
            raise ValueError(f"Invalid or duplicate GPU codec name: {name}")
        dtypes = spec.get("dtypes")
        found[name] = GpuCodec(name, folder, _validate(name, folder, declaration), tuple(dtypes) if dtypes else None)
    return found
