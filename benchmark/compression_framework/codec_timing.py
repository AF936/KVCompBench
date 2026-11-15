"""Codec-call timing with input preparation and output transfers excluded."""
import threading
import time

TIMING_VERSION = 'codec_call_v2'
_local = threading.local()
ALGORITHMS = {'lz4':'LZ4','snappy':'Snappy','gdeflate':'GDeflate','deflate':'Deflate','bitcomp':'Bitcomp','ans':'ANS','zstd':'Zstd','cascaded':'Cascaded','gzip':'Gzip'}

def gpu_call(function, stream, warmup=False):
    """Measure an API window on its actual CUDA stream, not a host enqueue."""
    import torch
    if warmup:
        function()
        stream.synchronize()
    start=torch.cuda.Event(enable_timing=True)
    end=torch.cuda.Event(enable_timing=True)
    start.record(stream)
    value=function()
    end.record(stream)
    end.synchronize()
    seconds=start.elapsed_time(end)/1000
    if seconds<=0:raise ValueError('GPU timer resolution is insufficient')
    return value,seconds

def encode_measured(route, raw, warmup=True):
    if not hasattr(_local,'engines'):_local.engines={};_local.warmed=set()
    if route.mode=='GPU-GPU':
        import torch
        from nvidia import nvcomp
        from batch_decode import load
        key=route.name
        if key not in _local.engines:
            stream=torch.cuda.Stream()
            codec=None if key=='splitzip' else nvcomp.Codec(algorithm=ALGORITHMS[key.removeprefix('nvcomp-')],cuda_stream=stream.cuda_stream)
            _local.engines[key]=(stream,codec)
        stream,codec=_local.engines[key]
        with torch.cuda.stream(stream):
            dtype=torch.bfloat16 if key=='splitzip' else torch.uint8
            tensor=torch.frombuffer(bytearray(raw),dtype=dtype).to('cuda',non_blocking=False)
            shape=(key,len(raw))
            if key=='splitzip':
                api=load(route.plugin.folder/'api.py');module=api.upstream()
                codec=module.ChunkLocalSplitZipGPU(device='cuda:0',chunk_size=1024)
                def call():
                    coverage=codec.calibrate(tensor)
                    return coverage,codec.encode(tensor)
                (coverage,encoded),seconds=gpu_call(call,stream,shape not in _local.warmed)
                header=dict(format='splitzip-packet-v1',n=encoded.n,n_esc=encoded.n_esc,chunk_size=encoded.chunk_size,codebook=codec.dec_lut.cpu().tolist(),calibration='per-block official calibrate; included in encode',coverage=coverage)
                frame=api.packet(header,[api.tensor_bytes(getattr(encoded,k)) for k in ('pk','sm','chunk_id','local_pos','esc_val')])
            else:
                source=nvcomp.as_array(tensor,cuda_stream=stream.cuda_stream)
                encoded,seconds=gpu_call(lambda:codec.encode(source),stream,shape not in _local.warmed)
                frame=bytes(encoded.cpu())
            _local.warmed.add(shape)
        return frame,seconds,None,None
    # Resolve/load the codec and warm it before measuring a host API invocation.
    from gpu_decode import backend,pack_frame
    key=(route.name,len(raw))
    if route.gpu:
        gpu=route.gpu
        if gpu.declaration['type']=='shared':
            function=backend(gpu.backend_id).cpu_encoder
            options=dict(gpu.declaration.get('encoder_options',{}))
            call=lambda:function(raw,options)
        else:
            module=gpu._load_module();function=getattr(module,gpu.declaration['cpu_encode'])
            call=lambda:function(raw)
    else:
        call=lambda:route.plugin.compress(raw,{'dtype':'bf16'})
    if warmup and key not in _local.warmed:call();_local.warmed.add(key)
    started=time.perf_counter();payload=call();ended=time.perf_counter()
    frame=pack_frame(route.gpu.backend_id,len(raw),bytes(payload)) if route.gpu else payload
    return frame,ended-started,started,ended
