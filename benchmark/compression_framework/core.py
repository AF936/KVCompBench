"""Plugin based, lossless block storage for raw KV tensor payloads."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import zipfile

from plugins import DEFAULT_COMPRESSORS, discover

DEFAULT_PLUGINS = DEFAULT_COMPRESSORS


def load_plugins(directory=DEFAULT_PLUGINS):
    from adapters.catalog import registered_plugins
    return discover(directory, registered_plugins())


def pack(source, target, codec='zlib', block_bytes=65536, metadata=None, plugin_dir=DEFAULT_PLUGINS):
    """Store raw tensor bytes; metadata describes their original tensor layout."""
    source, target = Path(source), Path(target)
    if type(block_bytes) is not int or not 0 < block_bytes <= 64 * 1024 * 1024:
        raise ValueError('block_bytes must be between 1 and 64 MiB')
    context = dict(metadata or {})
    json.dumps(context, allow_nan=False)
    available = load_plugins(plugin_dir)
    if codec not in available:
        raise ValueError(f'Unknown codec {codec!r}; add a codec.json package or run list')
    plugin = available[codec]
    from codec_manager import identity
    installation = identity(codec, gpu=getattr(plugin,'entry',{}).get('adapter')=='gpu-api', directory=plugin_dir)
    plugin.prepare(context) if hasattr(plugin, "prepare") else None
    if not plugin.supports(context):
        raise ValueError(f'{codec} does not support the supplied metadata')
    manifest = dict(installation=installation,format='kv-blocks-v1', codec=codec, codec_version=plugin.VERSION,
                    block_bytes=block_bytes, metadata=context, blocks=[], raw_bytes=0)
    # Exclusive creation protects both existing results and the source file.
    with target.open('xb') as output:
        with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            with source.open('rb') as handle:
                while raw := handle.read(block_bytes):
                    packet = plugin.compress(raw, dict(context))
                    if not isinstance(packet, bytes):
                        raise TypeError('compress must return bytes')
                    restored = plugin.decompress(packet, len(raw), dict(context))
                    if restored != raw:
                        raise ValueError('Plugin failed bit-exact roundtrip validation')
                    fallback = len(packet) >= len(raw)
                    payload = raw if fallback else packet
                    name = f'blocks/{len(manifest["blocks"]):012d}'
                    archive.writestr(name, payload)
                    manifest['blocks'].append(dict(name=name, raw_bytes=len(raw), stored_bytes=len(payload),
                        raw=fallback, sha256=hashlib.sha256(raw).hexdigest()))
                    manifest['raw_bytes'] += len(raw)
            archive.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False, allow_nan=False))
    return dict(raw_bytes=manifest['raw_bytes'], stored_bytes=target.stat().st_size,
                blocks=len(manifest['blocks']))


class Reader:
    """Read blocks independently, including in reverse or arbitrary order."""
    def __init__(self, path, plugin_dir=DEFAULT_PLUGINS):
        self.archive = zipfile.ZipFile(path)
        try:
            self.manifest = json.loads(self.archive.read('manifest.json'))
            m = self.manifest
            if m['format'] != 'kv-blocks-v1':
                raise ValueError('Unsupported container version')
            self.plugin = load_plugins(plugin_dir)[m['codec']]
            from codec_manager import identity
            self.installation = identity(m['codec'], gpu=getattr(self.plugin,'entry',{}).get('adapter')=='gpu-api', directory=plugin_dir)
            self.plugin.prepare(m['metadata']) if hasattr(self.plugin, 'prepare') else None
            if self.plugin.VERSION != m['codec_version']:
                raise ValueError('Codec version mismatch')
            if not self.plugin.supports(m['metadata']):
                raise ValueError('Unsupported metadata')
            if type(m['block_bytes']) is not int or not 0 < m['block_bytes'] <= 64 * 1024 * 1024:
                raise ValueError('Invalid block size')
            if len(set(self.archive.namelist())) != len(self.archive.namelist()):
                raise ValueError('Duplicate container entries')
            for i, b in enumerate(m['blocks']):
                if b['name'] != f'blocks/{i:012d}' or type(b['raw']) is not bool:
                    raise ValueError('Invalid block descriptor')
                if type(b['raw_bytes']) is not int or not 0 < b['raw_bytes'] <= m['block_bytes']:
                    raise ValueError('Invalid decoded size')
                if type(b['stored_bytes']) is not int or not 0 <= b['stored_bytes'] <= b['raw_bytes']:
                    raise ValueError('Invalid stored size')
                info = self.archive.getinfo(b['name'])
                if info.file_size != b['stored_bytes'] or info.compress_type != zipfile.ZIP_STORED:
                    raise ValueError('Invalid block storage')
            if sum(b['raw_bytes'] for b in m['blocks']) != m['raw_bytes']:
                raise ValueError('Invalid total size')
        except BaseException:
            self.archive.close()
            raise

    def read_block(self, index):
        b = self.manifest['blocks'][index]
        packet = self.archive.read(b['name'])
        raw = packet if b['raw'] else self.plugin.decompress(packet, b['raw_bytes'], dict(self.manifest['metadata']))
        if not isinstance(raw, bytes) or len(raw) != b['raw_bytes'] or hashlib.sha256(raw).hexdigest() != b['sha256']:
            raise ValueError('Block integrity check failed')
        return raw

    def close(self):
        self.archive.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def unpack(source, target, plugin_dir=DEFAULT_PLUGINS):
    with Reader(source, plugin_dir) as reader:
        with Path(target).open('xb') as output:
            for i in range(len(reader.manifest['blocks'])):
                output.write(reader.read_block(i))
