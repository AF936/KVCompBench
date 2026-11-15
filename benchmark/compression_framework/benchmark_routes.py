"""Benchmark dispatch: mature GPU decode when available, original CPU otherwise."""
from dataclasses import dataclass
from core import load_plugins
from gpu_codec import discover_gpu_codecs


@dataclass
class Route:
    name: str
    plugin: object
    gpu: object = None
    mode: str = 'CPU-CPU'
    declaration = {'type': 'shared'}

    @property
    def backend_id(self):
        return self.gpu.backend_id if self.gpu else self.mode + ':' + self.name

    def encode_cpu(self, raw, dtype='bf16'):
        return self.gpu.encode_cpu(raw, dtype=dtype) if self.gpu else self.plugin.compress(raw, {'dtype': dtype})

    def decode_gpu(self, frame, expected_bytes):
        return self.gpu.decode_gpu(frame, expected_bytes) if self.gpu else self.plugin.decompress(frame, expected_bytes, {'dtype': 'bf16'})


def discover_routes(directory):
    plugins = load_plugins(directory)
    gpu = discover_gpu_codecs(directory)
    routes = {}
    for name, plugin in plugins.items():
        if not plugin.supports({'dtype': 'bf16'}):
            continue
        if name.startswith('nvcomp-') or name == 'splitzip':
            routes[name] = Route(name, plugin, mode='GPU-GPU')
        elif name in gpu and (gpu[name].declaration['type'] == 'shared' and gpu[name].backend_id.startswith('nvcomp-') or name == 'openzl'):
            routes[name] = Route(name, plugin, gpu[name], 'CPU-GPU')
        else:
            routes[name] = Route(name, plugin)
    return routes
