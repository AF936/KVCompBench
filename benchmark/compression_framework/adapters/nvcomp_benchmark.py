"""Reusable nvCOMP API implementation; format parsing is a backend hook."""
from adapters.benchmark_cpu import CPUBackend
from gpu_decode import unpack_frame

def upload(parts):
    import torch
    offsets=[];cursor=0
    for part in parts:cursor=(cursor+15)//16*16;offsets.append(cursor);cursor+=len(part)
    host=torch.empty(cursor,dtype=torch.uint8,pin_memory=True)
    for offset,part in zip(offsets,parts):
        if part:host[offset:offset+len(part)].copy_(torch.frombuffer(bytearray(part),dtype=torch.uint8))
    return host,host.to('cuda',non_blocking=True),offsets

class Backend(CPUBackend):
    def __init__(self,route):
        super().__init__(route)
        import torch
        from nvidia import nvcomp
        self.stream=torch.cuda.Stream();self.codecs={};self.nvcomp=nvcomp
    def codec(self,algorithm,raw):
        key=(algorithm,raw)
        if key not in self.codecs:
            options=dict(algorithm=algorithm,cuda_stream=self.stream.cuda_stream)
            if raw:options['bitstream_kind']=self.nvcomp.BitstreamKind.RAW
            self.codecs[key]=self.nvcomp.Codec(**options)
        return self.codecs[key]
    def setup(self,sample):
        super().setup(sample)
        self.codec(self.route.options['algorithm'],self.route.options.get('raw',True))
    def prepare_encode(self,raw):
        if not self.route.mode.startswith('GPU-'):return raw
        import torch
        with torch.cuda.stream(self.stream):
            tensor=torch.frombuffer(bytearray(raw),dtype=torch.uint8).to('cuda')
            return tensor,self.nvcomp.as_array(tensor,cuda_stream=self.stream.cuda_stream)
    def encode(self,prepared):
        if not self.route.mode.startswith('GPU-'):return super().encode(prepared)
        return self.codec(self.route.options['algorithm'],False).encode(prepared[1])
    def finish_encode(self,value,size):
        if self.route.mode.startswith('GPU-'):return bytes(value.cpu())
        return super().finish_encode(value,size)
    def parse_frames(self,frames,sizes):
        if self.route.options.get('use_gpu_frame'):
            parsed=[unpack_frame(f) for f in frames]
            if any(ident!=self.route.gpu.backend_id or n!=size for (ident,n,p),size in zip(parsed,sizes)):raise ValueError('GPU frame mismatch')
            payloads=[p for ident,n,p in parsed]
        else:payloads=frames
        return payloads,list(sizes),[False]*len(sizes),[[i] for i in range(len(sizes))],[]
    def prepare_decode(self,frames,sizes):
        import torch
        payloads,raw_sizes,stored,groups,checks=self.parse_frames(frames,sizes)
        host,device,offsets=upload(payloads);pieces=[None]*len(payloads)
        indices=[i for i,n in enumerate(raw_sizes) if n and not stored[i]]
        for i,(is_stored,n) in enumerate(zip(stored,raw_sizes)):
            if is_stored or not n:pieces[i]=torch.empty(n,dtype=torch.uint8,device='cuda')
        sources=destinations=allocation=None
        if indices:
            codec=self.codec(self.route.options['algorithm'],self.route.options.get('raw',True))
            sources=self.nvcomp.as_arrays(device,[offsets[i] for i in indices],[len(payloads[i]) for i in indices],cuda_stream=self.stream.cuda_stream)
            dst_offsets=[];cursor=0
            for i in indices:cursor=(cursor+15)//16*16;dst_offsets.append(cursor);cursor+=raw_sizes[i]
            allocation=torch.empty(cursor,dtype=torch.uint8,device='cuda')
            destinations=self.nvcomp.as_arrays(allocation,dst_offsets,[raw_sizes[i] for i in indices],cuda_stream=self.stream.cuda_stream)
            for i,offset in zip(indices,dst_offsets):pieces[i]=allocation[offset:offset+raw_sizes[i]]
        def call():
            result=codec.decode(sources,out=destinations) if indices else None
            for i,(is_stored,n) in enumerate(zip(stored,raw_sizes)):
                if is_stored and n:pieces[i].copy_(device[offsets[i]:offsets[i]+n])
            return result
        return dict(call=call,finish=lambda result:[torch.cat([pieces[i] for i in group]) for group in groups],keep=(host,device,sources,destinations,allocation),checks=checks)
    def decode_signature(self,sizes):return tuple(sizes)
    def validate_decoded(self,parts,prepared):pass
