"""Upstream entropy codecs with raw/RLE tags and bounded Huff0 subframes."""
import ctypes as c
import struct
from adapters.paper_support import library

HUF_BLOCK_BYTES = 128 * 1024

def functions(algorithm):
    lib=library('fse');prefix='FSE' if algorithm=='fse' else 'HUF'
    enc=getattr(lib,prefix+'_compress');dec=getattr(lib,prefix+'_decompress')
    enc.argtypes=dec.argtypes=[c.c_void_p,c.c_size_t,c.c_void_p,c.c_size_t]
    enc.restype=dec.restype=c.c_size_t
    error=lib.FSE_isError;error.argtypes=[c.c_size_t];error.restype=c.c_uint
    return enc,dec,error

def compress(raw,metadata,algorithm='fse'):
    if algorithm=='huff0' and len(raw)>HUF_BLOCK_BYTES:
        frames=[]
        for start in range(0,len(raw),HUF_BLOCK_BYTES):
            part=raw[start:start+HUF_BLOCK_BYTES]
            frame=compress(part,metadata,algorithm)
            frames.append(struct.pack('<II',len(part),len(frame))+frame)
        return b'\3'+struct.pack('<I',len(frames))+b''.join(frames)
    if not raw:return b'\0'
    enc,_,error=functions(algorithm)
    source=c.create_string_buffer(raw);out=c.create_string_buffer(len(raw)+len(raw)//128+4096)
    n=enc(out,len(out),source,len(raw))
    if error(n):raise ValueError(f'{algorithm} compression error')
    if n==0:return b'\0'+raw
    if n==1:return b'\1'+raw[:1]
    return b'\2'+out.raw[:n]

def decompress(data,expected_bytes,metadata,algorithm='fse'):
    if not data:raise ValueError('Missing entropy tag')
    if data[0]==3:
        if algorithm!='huff0' or len(data)<5 or expected_bytes<=HUF_BLOCK_BYTES:
            raise ValueError('Invalid Huff0 subframe header')
        count=struct.unpack_from('<I',data,1)[0]
        if count!=(expected_bytes+HUF_BLOCK_BYTES-1)//HUF_BLOCK_BYTES:
            raise ValueError('Invalid Huff0 subframe count')
        offset=5;parts=[];remaining=expected_bytes
        for _ in range(count):
            if offset+8>len(data):raise ValueError('Truncated Huff0 subframe header')
            raw_size,frame_size=struct.unpack_from('<II',data,offset);offset+=8
            if raw_size!=min(remaining,HUF_BLOCK_BYTES) or frame_size<1 or offset+frame_size>len(data):
                raise ValueError('Invalid Huff0 subframe size')
            frame=data[offset:offset+frame_size]
            if frame[0]==3:raise ValueError('Nested Huff0 subframes are invalid')
            parts.append(decompress(frame,raw_size,metadata,algorithm))
            offset+=frame_size;remaining-=raw_size
        if offset!=len(data) or remaining:raise ValueError('Invalid Huff0 subframe payload')
        return b''.join(parts)
    if data[0]==0:raw=data[1:]
    elif data[0]==1:
        if len(data)!=2:raise ValueError('Invalid entropy RLE')
        raw=data[1:]*expected_bytes
    elif data[0]==2:
        _,dec,error=functions(algorithm);source=c.create_string_buffer(data[1:]);out=c.create_string_buffer(expected_bytes)
        n=dec(out,expected_bytes,source,len(data)-1)
        if error(n) or n!=expected_bytes:raise ValueError('Entropy decode failed')
        raw=out.raw
    else:raise ValueError('Invalid entropy tag')
    if len(raw)!=expected_bytes:raise ValueError('Entropy output length mismatch')
    return raw
