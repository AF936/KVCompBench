"""Upstream Stream VByte on uint32 bit patterns; zero-padded tail is length-checked."""
import ctypes as c
from adapters.paper_support import library


def compress(raw,metadata):
    if not raw:return b''
    lib=library('streamvbyte');n=(len(raw)+3)//4
    source=(c.c_uint32*n).from_buffer_copy(raw+b'\0'*((-len(raw))%4))
    target=c.create_string_buffer(5*n+128)
    f=lib.streamvbyte_encode;f.argtypes=[c.c_void_p,c.c_uint32,c.c_void_p];f.restype=c.c_size_t
    size=f(source,n,target)
    return target.raw[:size]


def decompress(data,expected_bytes,metadata):
    if not expected_bytes:
        if data:raise ValueError('Unexpected empty-block data')
        return b''
    lib=library('streamvbyte');n=(expected_bytes+3)//4;source=c.create_string_buffer(data,len(data)+64)
    valid=lib.streamvbyte_validate_stream;valid.argtypes=[c.c_void_p,c.c_size_t,c.c_uint32];valid.restype=c.c_bool
    if not valid(source,len(data),n):raise ValueError('Invalid Stream VByte stream')
    target=(c.c_uint32*n)();f=lib.streamvbyte_decode;f.argtypes=[c.c_void_p,c.c_void_p,c.c_uint32];f.restype=c.c_size_t
    used=f(source,target,n)
    if used!=len(data):raise ValueError('Trailing Stream VByte data')
    raw=bytes(target)
    if any(raw[expected_bytes:]):raise ValueError('Invalid tail padding')
    return raw[:expected_bytes]
