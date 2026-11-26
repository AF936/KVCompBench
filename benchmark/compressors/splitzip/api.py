"""Official SplitZip Triton codec with explicit, independently decodable packets."""
from functools import lru_cache
import importlib.util
import os
from pathlib import Path
import sys
from adapters.paper_support import packet,unpacket

@lru_cache(maxsize=2)
def upstream(filename='codec_gpu.py'):
    from codec_manager import source_file
    path=source_file('splitzip',filename)
    spec=importlib.util.spec_from_file_location('kv_official_splitzip_'+path.stem,path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def runtime():
    if not os.environ.get('CUDA_VISIBLE_DEVICES','').strip():raise RuntimeError('An administrator-assigned CUDA_VISIBLE_DEVICES is required')
    cache=Path(sys.prefix)/'tmp'
    for key,folder in [('TRITON_CACHE_DIR','triton'),('CUDA_CACHE_PATH','cuda'),('TORCHINDUCTOR_CACHE_DIR','torchinductor')]:
        path=cache/folder;path.mkdir(parents=True,exist_ok=True);os.environ.setdefault(key,str(path))
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable in the selected environment')
    return torch


def tensor_bytes(value):return value.detach().cpu().contiguous().view(__import__('torch').uint8).numpy().tobytes()


def compress(raw,metadata):
    if metadata.get('dtype') in ('fp8_e4m3fn','fp8_e5m2'):
        from benchmark_routes import module_at
        return module_at(Path(__file__).with_name('fp8_backend.py')).compress(sys.modules[__name__],raw,metadata)
    if metadata.get('dtype')!='bf16' or len(raw)%2:raise ValueError('SplitZip requires native BF16 bytes')
    torch=runtime();module=upstream();chunk=int(metadata.get('splitzip_chunk_size',1024))
    codec=module.ChunkLocalSplitZipGPU(device='cuda:0',chunk_size=chunk)
    tensor=torch.frombuffer(bytearray(raw),dtype=torch.bfloat16).to('cuda:0')
    # The calibration pass is part of compression and is explicitly labelled.
    coverage=codec.calibrate(tensor)
    encoded=codec.encode(tensor)
    book=codec.dec_lut.cpu().tolist()
    header=dict(format='splitzip-packet-v1',n=encoded.n,n_esc=encoded.n_esc,chunk_size=encoded.chunk_size,
                codebook=book,calibration='per-block official calibrate; included in encode',coverage=coverage)
    return packet(header,[tensor_bytes(getattr(encoded,k)) for k in ('pk','sm','chunk_id','local_pos','esc_val')])


def decompress(data,expected_bytes,metadata):
    if metadata.get('dtype') in ('fp8_e4m3fn','fp8_e5m2'):
        from benchmark_routes import module_at
        return module_at(Path(__file__).with_name('fp8_backend.py')).decompress(sys.modules[__name__],data,expected_bytes,metadata)
    torch=runtime();module=upstream();header,parts=unpacket(data)
    n,esc,chunk=header['n'],header['n_esc'],header['chunk_size']
    if header['format']!='splitzip-packet-v1' or type(n)is not int or n*2!=expected_bytes or type(esc)is not int or not 0<=esc<=n:
        raise ValueError('Invalid SplitZip dimensions')
    if len(parts)!=5 or [len(b) for b in parts]!=[(n+1)//2,n,4*esc,2*esc,esc]:raise ValueError('Invalid SplitZip buffer sizes')
    book=header['codebook']
    if len(book)!=16 or any(type(x)is not int or not 0<=x<=255 for x in book):raise ValueError('Invalid SplitZip codebook')
    codec=module.ChunkLocalSplitZipGPU(device='cuda:0',chunk_size=chunk)
    codec.dec_lut=torch.tensor(book,dtype=torch.uint8,device='cuda:0')
    # Decoder uses dec_lut only; initialize the remaining documented ready-state arrays.
    codec.enc_lut=torch.zeros(256,dtype=torch.uint8,device='cuda:0')
    codec.enc_lut_marked=torch.full((256,),16,dtype=torch.uint8,device='cuda:0')
    codec.common_lut=torch.zeros(256,dtype=torch.uint8,device='cuda:0')
    def restore(b,dtype):
        return torch.frombuffer(bytearray(b),dtype=dtype).to('cuda:0') if b else torch.empty(0,dtype=dtype,device='cuda:0')
    pk,sm,chunk_id,local_pos,esc_val=[restore(b,d) for b,d in zip(parts,[torch.uint8,torch.uint8,torch.int32,torch.uint16,torch.uint8])]
    encoded=module.ChunkLocalGPUEncoded(pk,sm,torch.empty(0,dtype=torch.int32,device='cuda:0'),torch.empty(0,dtype=torch.int64,device='cuda:0'),chunk_id,local_pos,esc_val,n,esc,chunk)
    return tensor_bytes(codec.decode(encoded))
