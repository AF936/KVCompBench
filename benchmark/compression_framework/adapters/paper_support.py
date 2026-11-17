"""Shared ABI and framing utilities for upstream codec adapters."""
import ctypes
from functools import lru_cache
import json
from pathlib import Path
import struct
import sys

@lru_cache(maxsize=None)
def library(name):
    path=Path(sys.prefix)/'native'/name/('codec.dll' if sys.platform=='win32' else 'libcodec.so')
    if not path.is_file():raise ImportError(f'Build {name} in the selected environment: bash setup_paper_codecs.sh')
    return ctypes.CDLL(str(path))


def packet(header, buffers):
    header=dict(header,lengths=[len(b) for b in buffers])
    encoded=json.dumps(header,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return struct.pack('<I',len(encoded))+encoded+b''.join(buffers)


def unpacket(data,maximum=64*1024*1024):
    if len(data)<4:raise ValueError('Truncated packet')
    size=struct.unpack_from('<I',data)[0]
    if not 0<size<=1024*1024 or 4+size>len(data):raise ValueError('Invalid header size')
    header=json.loads(data[4:4+size]);offset=4+size;parts=[]
    for n in header['lengths']:
        if type(n)is not int or not 0<=n<=maximum or offset+n>len(data):raise ValueError('Invalid segment size')
        parts.append(data[offset:offset+n]);offset+=n
    if offset!=len(data):raise ValueError('Trailing packet data')
    return header,parts
