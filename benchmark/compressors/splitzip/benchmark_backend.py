"""Frozen dataset calibration and merged official-kernel SplitZip batch decoding."""
from adapters.benchmark_cpu import CPUBackend
from adapters.nvcomp_benchmark import upload
from benchmark_routes import module_at
from adapters.paper_support import packet,unpacket

class Backend(CPUBackend):
    def __init__(self,route):
        super().__init__(route)
        import torch
        self.torch=torch;self.stream=torch.cuda.Stream();self.api=module_at(route.folder/'api.py');self.module=self.api.upstream()
        self.chunk=int(route.options.get('chunk_size',1024));self.codec=self.module.ChunkLocalSplitZipGPU(device='cuda:0',chunk_size=self.chunk)
    def setup(self,sample):
        with self.torch.cuda.stream(self.stream):
            tensor=self.torch.frombuffer(bytearray(sample),dtype=self.torch.bfloat16).to('cuda')
            self.coverage=self.codec.calibrate(tensor)
        self.stream.synchronize();self.book=self.codec.dec_lut.cpu().tolist()
    def prepare_encode(self,raw):
        with self.torch.cuda.stream(self.stream):return self.torch.frombuffer(bytearray(raw),dtype=self.torch.bfloat16).to('cuda')
    def encode(self,tensor):
        with self.torch.cuda.stream(self.stream):return self.codec.encode(tensor)
    def finish_encode(self,value,size):
        header=dict(format='splitzip-packet-v1',n=value.n,n_esc=value.n_esc,chunk_size=value.chunk_size,codebook=self.book,calibration='frozen first page; excluded from codec timing',coverage=self.coverage)
        return packet(header,[self.api.tensor_bytes(getattr(value,k)) for k in ('pk','sm','chunk_id','local_pos','esc_val')])
    def prepare_decode(self,frames,sizes):
        torch=self.torch;packets=[unpacket(f) for f in frames];buffers=[]
        for (h,parts),size in zip(packets,sizes):
            n=h['n'];esc=h['n_esc']
            if h['format']!='splitzip-packet-v1' or n*2!=size or not 0<=esc<=n or len(parts)!=5 or [len(p) for p in parts]!=[(n+1)//2,n,4*esc,2*esc,esc]:raise ValueError('Invalid SplitZip frame')
            if h['codebook']!=self.book or h['chunk_size']!=self.chunk:raise ValueError('SplitZip batch codebook/chunk mismatch')
            buffers.extend(parts)
        # Padding is outside timing; it permits arbitrary tail pages while retaining
        # every page's independent escape positions and a single official decode call.
        host,device,offsets=upload(buffers);pk=[];sm=[];cid=[];pos=[];escapes=[];segments=[];base=0
        for i,((h,parts),size) in enumerate(zip(packets,sizes)):
            views=[device[o:o+len(p)] for o,p in zip(offsets[i*5:i*5+5],parts)]
            n=h['n'];padded=(n+self.chunk-1)//self.chunk*self.chunk
            pk.append(torch.nn.functional.pad(views[0],(0,padded//2-len(parts[0]))))
            sm.append(torch.nn.functional.pad(views[1],(0,padded-n)))
            cid.append(views[2].view(torch.int32)+base//self.chunk);pos.append(views[3]);escapes.append(views[4]);segments.append((base*2,size));base+=padded
        merged=self.module.ChunkLocalGPUEncoded(torch.cat(pk),torch.cat(sm),torch.empty(0,dtype=torch.int32,device='cuda'),torch.empty(0,dtype=torch.int64,device='cuda'),torch.cat(cid),torch.cat(pos).view(torch.uint16),torch.cat(escapes),base,sum(h['n_esc'] for h,p in packets),self.chunk)
        return dict(call=lambda:self.codec.decode(merged).view(torch.uint8),finish=lambda out:[out[start:start+size] for start,size in segments],keep=(host,device,merged))
    def decode_signature(self,sizes):return tuple(sizes)
    def validate_decoded(self,parts,prepared):pass
