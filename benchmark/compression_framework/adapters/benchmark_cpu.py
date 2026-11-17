"""Generic CPU plugin API and compatible decoder resolution."""

class CPUBackend:
    stream=None
    def __init__(self,route):
        self.route=route;self.setup_seconds=0
        self.thread_safe=route.options.get('thread_safe',True)
    def setup(self,sample):
        if self.route.gpu and self.route.gpu.declaration['type']=='module':self.route.gpu._load_module()
        elif hasattr(self.route.plugin,'prepare'):self.route.plugin.prepare({'dtype':'bf16'})
    def prepare_encode(self,raw):return raw
    def encode(self,raw):
        gpu=self.route.gpu
        if gpu:
            from gpu_decode import backend
            if gpu.declaration['type']=='shared':return backend(gpu.backend_id).cpu_encoder(raw,dict(gpu.declaration.get('encoder_options',{})))
            return getattr(gpu._load_module(),gpu.declaration['cpu_encode'])(raw)
        return self.route.plugin.compress(raw,{'dtype':'bf16'})
    def finish_encode(self,value,size):
        if self.route.gpu:
            from gpu_decode import pack_frame
            return pack_frame(self.route.gpu.backend_id,size,bytes(value))
        return bytes(value)
    def cpu_decoder(self):
        if self.route.mode.endswith('-CPU'):
            return lambda frame,size:self.route.plugin.decompress(frame,size,{'dtype':'bf16'})
        from cpu_decode_registry import resolve_decoder
        return resolve_decoder(self.route)[0]
