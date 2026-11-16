"""OpenZL frame parsing and checksum validation around shared nvCOMP batching."""
from adapters.nvcomp_benchmark import Backend as NVBackend
from gpu_decode import unpack_frame
class Backend(NVBackend):
    def parse_frames(self,frames,sizes):
        import zstandard
        api=self.route.gpu._load_module();payloads=[];raw_sizes=[];stored=[];groups=[];checks=[]
        for frame,size in zip(frames,sizes):
            ident,declared,payload=unpack_frame(frame)
            if ident!=self.route.gpu.backend_id or declared!=size:raise ValueError('OpenZL frame mismatch')
            _,_,pieces=api.parse(payload,size,api.xxhash.xxh3_64_intdigest);group=[];total=0
            for encoded,check,is_stored in pieces:
                p=encoded if is_stored else api.standard_zstd_payload(encoded,27)
                n=len(p) if is_stored else zstandard.frame_content_size(p)
                if n<0:raise ValueError('Unknown OpenZL size')
                group.append(len(payloads));payloads.append(p);raw_sizes.append(n);stored.append(is_stored);checks.append((n,check[0]));total+=n
            if total!=size:raise ValueError('OpenZL size mismatch')
            groups.append(group)
        return payloads,raw_sizes,stored,groups,checks
    def validate_decoded(self,parts,prepared):
        data=b''.join(parts);offset=0;api=self.route.gpu._load_module()
        for size,expected in prepared['checks']:
            if expected is not None and api.xxhash.xxh3_64_intdigest(data[offset:offset+size])&0xffffffff!=expected:raise ValueError('OpenZL checksum mismatch')
            offset+=size
