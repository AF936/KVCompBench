"""Adapters to callable codec APIs; no project compression algorithm implementations."""
from functools import lru_cache
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parents[1]
WIDTHS = {'bytes':1,'uint8':1,'bf16':2,'fp16':2,'fp32':4,'fp64':8,'fp8_e4m3fn':1,'fp8_e5m2':1}


class Codec:
    def __init__(self,entry):
        self.entry=entry
        self.NAME=entry['name']
        self.VERSION='api-v3'
        self.fingerprint=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]

    def supports(self,metadata):
        dtype=metadata.get('dtype','bytes')
        if dtype not in WIDTHS:return False
        if self.NAME=='fpzip32':return dtype=='fp32'
        if self.NAME=='fpzip64':return dtype=='fp64'
        return True

    def prepare(self,metadata):
        if not self.supports(metadata):raise NotImplementedError(self.NAME+': unsupported dtype')
        self.engine(metadata)

    @lru_cache(maxsize=8)
    def _engine(self,metadata_json):
        name=self.NAME
        if name.startswith('nvcomp-'):
            if not os.environ.get('CUDA_VISIBLE_DEVICES', '').strip():
                raise RuntimeError('Set CUDA_VISIBLE_DEVICES to the GPU assigned by the administrator before GPU use')
            from nvidia import nvcomp
            algorithm=self.entry['algorithm']
            codec=nvcomp.Codec(algorithm=algorithm,**self.entry.get('codec_options',{}))
            # Native framed format includes the metadata required by the official decoder.
            def encode(raw):return bytes(codec.encode(nvcomp.as_array(raw)).cpu())
            def decode(packet,n):return bytes(codec.decode(nvcomp.as_array(packet)).cpu())
            return encode,decode,'nvcomp/'+nvcomp.__version__+'/'+algorithm
        if name in ('identity','gzip','bz2','zlib','lzma'):
            import bz2,gzip,lzma,platform,zlib
            if name=='identity':return bytes,lambda b,n:bytes(b),'raw-v1'
            module={'gzip':gzip,'bz2':bz2,'zlib':zlib,'lzma':lzma}[name]
            return module.compress,lambda b,n:module.decompress(b),platform.python_version()
        base, _, level = name.partition(':')
        if base == 'zstd':
            import zstandard
            enc = zstandard.ZstdCompressor(level=int(level or 3)).compress
            dec = zstandard.ZstdDecompressor().decompress
            return enc,lambda b,n:dec(b,max_output_size=n),importlib.metadata.version('zstandard')
        if base == 'lz4':
            import lz4.frame
            return lz4.frame.compress,lambda b,n:lz4.frame.decompress(b),importlib.metadata.version('lz4')
        if base == 'snappy':
            import snappy
            return snappy.compress,lambda b,n:snappy.decompress(b),importlib.metadata.version('python-snappy')
        if base in ('zlib','lzma'):
            import zlib,lzma
            enc = (lambda b:zlib.compress(b,int(level))) if base == 'zlib' else (lambda b:lzma.compress(b,preset=int(level)))
            dec = zlib.decompress if base == 'zlib' else lzma.decompress
            return enc,lambda b,n:dec(b),sys.version.split()[0]
        if base in ('fpzip32','fpzip64'):
            import numpy as np
            import fpzip
            dtype = '<f4' if base == 'fpzip32' else '<f8'
            width = 4 if base == 'fpzip32' else 8
            def enc(raw):
                if len(raw) % width:raise ValueError('Unaligned native fpzip input')
                values = np.frombuffer(raw,dtype=dtype).copy().reshape(1,1,1,-1)
                return fpzip.compress(values,precision=0,order='C')
            def dec(packet,n):
                return fpzip.decompress(packet,order='C').astype(dtype,copy=False).tobytes()
            return enc,dec,importlib.metadata.version('fpzip')
        raise ValueError('Unknown API codec: '+name)

    def engine(self,metadata):
        engine=self._engine(json.dumps(metadata,sort_keys=True))
        fingerprint=self.fingerprint
        self.VERSION=f'api-v3/{fingerprint}/{engine[2]}'
        return engine

    def compress(self,raw,metadata):return bytes(self.engine(metadata)[0](raw))

    def decompress(self,packet,expected_bytes,metadata):
        raw=bytes(self.engine(metadata)[1](packet,expected_bytes))
        if len(raw)!=expected_bytes:raise ValueError('Decoded length mismatch')
        return raw


def registered_plugins():
    entries=json.loads((HERE/'registry.json').read_text())['codecs']
    plugins={entry['name']:Codec(entry) for entry in entries}
    if len(plugins)!=len(entries):raise ValueError('Duplicate codec names')
    return plugins
