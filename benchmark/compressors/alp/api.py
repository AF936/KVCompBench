"""Frame the upstream ALP/ALP_RD calls; original IEEE bits are retained or stored raw."""
import ctypes as c
from adapters.paper_support import library


def compress(raw,metadata):
    width={'fp32':4,'fp64':8}[metadata['dtype']]
    if len(raw)%width:raise ValueError('ALP native dtype alignment')
    lib=library('alp');f=lib.kv_alp_encode;f.argtypes=[c.c_void_p,c.c_size_t,c.c_int,c.c_void_p,c.c_size_t];f.restype=c.c_int64
    src=c.create_string_buffer(raw);out=c.create_string_buffer(max(65536,len(raw)*8+65536));size=f(src,len(raw),width,out,len(out))
    if size<0:raise ValueError(f'Upstream ALP encode failed: {size}')
    return out.raw[:size]


def decompress(data,expected_bytes,metadata):
    width={'fp32':4,'fp64':8}[metadata['dtype']]
    lib=library('alp');f=lib.kv_alp_decode;f.argtypes=[c.c_void_p,c.c_size_t,c.c_int,c.c_void_p,c.c_size_t];f.restype=c.c_int64
    src=c.create_string_buffer(data);out=c.create_string_buffer(expected_bytes);size=f(src,len(data),width,out,expected_bytes)
    if size!=expected_bytes:raise ValueError('Upstream ALP decode failed')
    return out.raw
