"""Common timing boundaries. Codec setup and data movement are reported separately."""
import time
TIMING_VERSION='codec_call_v3_kv_pages_shared_encode'

def gpu_call(function,stream,warmup=False):
    import torch
    if warmup:function();stream.synchronize()
    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    start.record(stream);value=function();end.record(stream);end.synchronize()
    return value,max(start.elapsed_time(end)/1000,1e-12)

class Encoder:
    def __init__(self,runtime,sample):
        self.runtime=runtime;self.warmed=set()
        started=time.perf_counter();runtime.setup(sample)
        if runtime.stream is not None:runtime.stream.synchronize()
        self.setup_seconds=time.perf_counter()-started
    def measure(self,raw):
        runtime=self.runtime
        def run():
            prepared=runtime.prepare_encode(raw)
            if runtime.route.mode.startswith('CPU-'):
                t=time.perf_counter();value=runtime.encode(prepared);seconds=time.perf_counter()-t
            else:
                runtime.stream.synchronize()
                value,seconds=gpu_call(lambda:runtime.encode(prepared),runtime.stream)
            return runtime.finish_encode(value,len(raw)),seconds
        if len(raw) not in self.warmed:
            run();self.warmed.add(len(raw))
        started=time.perf_counter();frame,seconds=run();ended=time.perf_counter()
        return frame,seconds,ended-started,started,ended

    def measure_many(self,raws,pool,workers):
        if workers==1 or not self.runtime.route.mode.startswith('CPU-'):
            return [self.measure(raw) for raw in raws]
        results=[]
        for start in range(0,len(raws),workers):
            wave=raws[start:start+workers]
            for raw in wave:
                if len(raw) not in self.warmed:self.measure(raw)
            page_start=time.perf_counter()
            prepared=[self.runtime.prepare_encode(raw) for raw in wave]
            def call(value):
                t0=time.perf_counter();encoded=self.runtime.encode(value);t1=time.perf_counter()
                return encoded,t0,t1
            measured=list(pool.map(call,prepared))
            elapsed=max(t1 for value,t0,t1 in measured)-min(t0 for value,t0,t1 in measured)
            frames=[self.runtime.finish_encode(value,len(raw)) for raw,(value,t0,t1) in zip(wave,measured)]
            page_seconds=time.perf_counter()-page_start;total=sum(map(len,wave))
            results.extend((frame,elapsed*len(raw)/total,page_seconds*len(raw)/total,t0,t1) for raw,frame,(value,t0,t1) in zip(wave,frames,measured))
        return results
