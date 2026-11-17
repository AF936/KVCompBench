"""Generic GPU batch scheduling. Backend owns parsing and decode implementation."""
import time
from codec_timing import gpu_call

class BatchDecoder:
    def __init__(self,runtime):self.runtime=runtime;self.warmed=set();self.page_seconds=0
    def submit(self,frames,sizes):
        import torch
        runtime=self.runtime;stream=runtime.stream
        shape=runtime.decode_signature(sizes)
        if shape not in self.warmed:
            with torch.cuda.stream(stream):
                warm=runtime.prepare_decode(frames,sizes);warm['call']();stream.synchronize()
            self.warmed.add(shape)
        started=time.perf_counter()
        with torch.cuda.stream(stream):
            prepared=runtime.prepare_decode(frames,sizes)
            stream.synchronize()
            value,seconds=gpu_call(prepared['call'],stream)
            outputs=prepared['finish'](value)
            packed=torch.cat(outputs)
            host=torch.empty(packed.numel(),dtype=torch.uint8,pin_memory=True)
            host.copy_(packed,non_blocking=True);stream.synchronize()
        raw=host.numpy().tobytes();parts=[];offset=0
        for size in sizes:parts.append(raw[offset:offset+size]);offset+=size
        if offset!=len(raw):raise ValueError('Batch output size mismatch')
        runtime.validate_decoded(parts,prepared)
        self.page_seconds=time.perf_counter()-started
        self.result=(parts,seconds,runtime.route.options['batch_kind'])
        return self
    def finish(self):return self.result
