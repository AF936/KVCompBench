"""Batch GPU upload, decode and a single packed download per batch."""
import importlib.util
from pathlib import Path
import torch
from gpu_decode import unpack_frame, backend
from codec_timing import gpu_call

_modules={}
def load(path):
    path=Path(path)
    if path not in _modules:
        spec=importlib.util.spec_from_file_location('batch_'+path.parent.name,path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        _modules[path]=module
    return _modules[path]

def upload(parts, alignment=16):
    offsets=[];size=0
    for p in parts:
        size=(size+alignment-1)//alignment*alignment;offsets.append(size);size+=len(p)
    host=torch.empty(size,dtype=torch.uint8,pin_memory=True)
    for offset,p in zip(offsets,parts):
        if p:host[offset:offset+len(p)].copy_(torch.frombuffer(bytearray(p),dtype=torch.uint8))
    device=host.to('cuda',non_blocking=True)
    return host,device,offsets

class BatchDecoder:
    def __init__(self,route):
        self.route=route
        self.stream=torch.cuda.Stream()
        self.codecs={}
        self.warmed=set()

    def codec(self,algorithm,raw):
        from nvidia import nvcomp
        key=(algorithm,raw)
        if key not in self.codecs:
            options=dict(algorithm=algorithm,cuda_stream=self.stream.cuda_stream)
            if raw:options['bitstream_kind']=nvcomp.BitstreamKind.RAW
            self.codecs[key]=nvcomp.Codec(**options)
        return self.codecs[key]

    def submit(self,frames,sizes):
        from nvidia import nvcomp
        keep=[];checks=[]
        with torch.cuda.stream(self.stream):
            if self.route.name=='splitzip':
                api=load(self.route.plugin.folder/'api.py')
                module=api.upstream()
                from adapters.paper_support import unpacket
                packets=[unpacket(f) for f in frames]
                buffers=[]
                for (header,parts),size in zip(packets,sizes):
                    n=header['n'];esc=header['n_esc']
                    if header['format']!='splitzip-packet-v1' or n*2!=size or not 0<=esc<=n or len(parts)!=5 or [len(p) for p in parts]!=[(n+1)//2,n,4*esc,2*esc,esc]:
                        raise ValueError('Invalid SplitZip packet')
                    book=header['codebook']
                    if len(book)!=16 or any(type(v)is not int or not 0<=v<=255 for v in book):raise ValueError('Invalid SplitZip codebook')
                    buffers.extend(parts+[bytes(book)])
                host,device,offsets=upload(buffers);keep.extend((host,device))
                tasks=[]
                for i,((header,parts),size) in enumerate(zip(packets,sizes)):
                    views=[device[o:o+len(p)] for o,p in zip(offsets[i*6:i*6+6],buffers[i*6:i*6+6])]
                    pk,sm,cid,pos,esc,book=views
                    codec=module.ChunkLocalSplitZipGPU(device='cuda:0',chunk_size=header['chunk_size'])
                    codec.dec_lut=book;codec.enc_lut=torch.zeros(256,dtype=torch.uint8,device='cuda')
                    codec.enc_lut_marked=torch.full((256,),16,dtype=torch.uint8,device='cuda');codec.common_lut=codec.enc_lut
                    encoded=module.ChunkLocalGPUEncoded(pk,sm,torch.empty(0,dtype=torch.int32,device='cuda'),torch.empty(0,dtype=torch.int64,device='cuda'),cid.view(torch.int32),pos.view(torch.uint16),esc,header['n'],header['n_esc'],header['chunk_size'])
                    tasks.append((codec,encoded));keep.extend((codec,encoded))
                shape=('splitzip',tuple(sizes))
                outputs,seconds=gpu_call(lambda:[codec.decode(encoded).view(torch.uint8) for codec,encoded in tasks],self.stream,shape not in self.warmed)
                self.warmed.add(shape)
                kind='official-kernel-queue'
            else:
                if self.route.name=='openzl':
                    api=self.route.gpu._load_module()
                    import zstandard
                    payloads=[];raw_sizes=[];stored=[];groups=[]
                    for frame,size in zip(frames,sizes):
                        _,declared,payload=unpack_frame(frame)
                        if declared!=size:raise ValueError('Frame size mismatch')
                        _,_,pieces=api.parse(payload,size,api.xxhash.xxh3_64_intdigest)
                        group=[];total=0
                        for encoded,check,is_stored in pieces:
                            p=encoded if is_stored else api.standard_zstd_payload(encoded,27)
                            n=len(p) if is_stored else zstandard.frame_content_size(p)
                            if n<0:raise ValueError('Unknown OpenZL size')
                            group.append(len(payloads));payloads.append(p);raw_sizes.append(n);stored.append(is_stored);checks.append((n,check[0]));total+=n
                        if total!=size:raise ValueError('OpenZL size mismatch')
                        groups.append(group)
                    algorithm='Zstd';raw=True
                elif self.route.gpu:
                    parsed=[unpack_frame(f) for f in frames]
                    if any(n!=size or ident!=self.route.backend_id for (ident,n,p),size in zip(parsed,sizes)):raise ValueError('GPU frame mismatch')
                    payloads=[p for _,_,p in parsed];raw_sizes=sizes;stored=[False]*len(sizes)
                    algorithm=backend(self.route.backend_id).nvcomp_algorithm;raw=True;groups=[[i] for i in range(len(sizes))]
                else:
                    payloads=frames;raw_sizes=sizes;stored=[False]*len(sizes);raw=False
                    algorithm={'gdeflate':'GDeflate','bitcomp':'Bitcomp','ans':'ANS','zstd':'Zstd','cascaded':'Cascaded','gzip':'Gzip','lz4':'LZ4','snappy':'Snappy','deflate':'Deflate'}[self.route.name.removeprefix('nvcomp-')]
                    groups=[[i] for i in range(len(sizes))]
                host,device,offsets=upload(payloads);keep.extend((host,device))
                pieces=[None]*len(payloads)
                indices=[i for i,n in enumerate(raw_sizes) if n and not stored[i]]
                for i,(is_stored,n) in enumerate(zip(stored,raw_sizes)):
                    if is_stored or not n:pieces[i]=torch.empty(n,dtype=torch.uint8,device='cuda')
                sources=destinations=allocation=None
                if indices:
                    codec=self.codec(algorithm,raw)
                    sources=nvcomp.as_arrays(device,[offsets[i] for i in indices],[len(payloads[i]) for i in indices],cuda_stream=self.stream.cuda_stream)
                    destination_offsets=[];cursor=0
                    # Align all destinations, including nonstandard final pieces.
                    for i in indices:cursor=(cursor+15)//16*16;destination_offsets.append(cursor);cursor+=raw_sizes[i]
                    allocation=torch.empty(cursor,dtype=torch.uint8,device='cuda')
                    destinations=nvcomp.as_arrays(allocation,destination_offsets,[raw_sizes[i] for i in indices],cuda_stream=self.stream.cuda_stream)
                    for i,offset in zip(indices,destination_offsets):pieces[i]=allocation[offset:offset+raw_sizes[i]]
                def call():
                    result=codec.decode(sources,out=destinations) if indices else None
                    for i,(is_stored,n) in enumerate(zip(stored,raw_sizes)):
                        if is_stored and n:pieces[i].copy_(device[offsets[i]:offsets[i]+n])
                    return result
                shape=(algorithm,raw,tuple(raw_sizes),tuple(stored))
                result,seconds=gpu_call(call,self.stream,shape not in self.warmed)
                self.warmed.add(shape)
                keep.extend((sources,destinations,result,allocation))
                outputs=[torch.cat([pieces[i] for i in group]) for group in groups]
                kind='nvcomp-batch'
            packed=torch.cat(outputs)
            host_result=torch.empty(packed.numel(),dtype=torch.uint8,pin_memory=True)
            host_result.copy_(packed,non_blocking=True)
            event=torch.cuda.Event();event.record(self.stream)
            keep.extend((outputs,packed,host_result))
        return Job(event,host_result,sizes,keep,seconds,kind,self.route,checks)

class Job:
    def __init__(self,event,host,sizes,keep,seconds,kind,route,checks):
        self.event,self.host,self.sizes,self.keep,self.seconds,self.kind,self.route,self.checks=event,host,sizes,keep,seconds,kind,route,checks
    def finish(self):
        self.event.synchronize()
        all_bytes=self.host.numpy().tobytes();parts=[];offset=0
        for size in self.sizes:parts.append(all_bytes[offset:offset+size]);offset+=size
        if offset!=len(all_bytes):raise ValueError('Batch output size mismatch')
        if self.checks:
            api=self.route.gpu._load_module();offset=0
            # OpenZL pieces remain in their original frame/stream order.
            for size,expected in self.checks:
                if expected is not None and api.xxhash.xxh3_64_intdigest(all_bytes[offset:offset+size]) & 0xffffffff != expected:
                    raise ValueError('OpenZL content checksum mismatch')
                offset+=size
        return parts,self.seconds,self.kind
