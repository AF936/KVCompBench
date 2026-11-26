"""Official SplitZip FP8 kernels in independently decodable KV page packets."""
import struct
from adapters.benchmark_cpu import CPUBackend
from adapters.nvcomp_benchmark import upload
from adapters.paper_support import packet, unpacket
from benchmark_routes import module_at

FORMAT = "splitzip-fp8-packet-v1"
FORMATS = {"fp8_e4m3fn": ("e4m3", 3), "fp8_e5m2": ("e5m2", 4)}


def make_codec(api, dtype):
    module = api.upstream("codec_fp8_gpu.py")
    fmt, bits = FORMATS[dtype]
    return module, module.SplitZipFP8GPU(fmt, bits, device="cuda:0", force_lut=True)


def frame(api, codec, value, scale):
    parts = [value.p32, value.p16, value.p8, value.counts.to(__import__('torch').uint16),
             value.esc, value.tail]
    header = dict(format=FORMAT, dtype=codec.fmt, code_bits=codec.cb, n=value.n,
                  n_esc=value.n_esc, codebook=codec.codebook)
    return packet(header, [b"" if x is None else api.tensor_bytes(x) for x in parts]+[scale])


def parse(module, codec, data, raw_size):
    h, parts = unpacket(data)
    n, escapes, book = h["n"], h["n_esc"], h["codebook"]
    if (h["format"], h["dtype"], h["code_bits"]) != (FORMAT, codec.fmt, codec.cb):
        raise ValueError("SplitZip FP8 frame format or dtype mismatch")
    if type(n) is not int or not 0 < n <= raw_size or type(escapes) is not int or not 0 <= escapes <= n:
        raise ValueError("Invalid SplitZip FP8 frame dimensions")
    if (len(book) != codec.K or any(type(x) is not int or not 0 <= x < 1 << codec.exp_bits for x in book)
            or sorted(set(book)) != book or codec.codebook is not None and book != codec.codebook):
        raise ValueError("Invalid SplitZip FP8 codebook")
    chunks = codec._nch(n)
    lanes = chunks * module.LANES
    expected = [4*lanes, 2*lanes if codec.plane16 else 0, lanes if codec.plane8 else 0,
                2*chunks, 2*escapes, n-chunks*module.CHUNK, raw_size-n]
    if list(map(len, parts)) != expected:
        raise ValueError("Invalid SplitZip FP8 segment lengths")
    counts = [count for (count,) in struct.iter_unpack("<H", parts[3])]
    if sum(counts) != escapes or any(count > module.CHUNK for count in counts):
        raise ValueError("Invalid SplitZip FP8 escape counts")
    return h, parts


def device_encoded(torch, module, codec, h, views):
    p32, p16, p8, counts, esc, tail, scale = views
    counts = counts.view(torch.uint16).to(torch.int32)
    starts = (counts.cumsum(0)-counts).to(torch.int32)
    encoded = module.FP8Encoded(p32.view(torch.int32),
                                p16.view(torch.int16) if codec.plane16 else None,
                                p8 if codec.plane8 else None, counts, starts,
                                esc.view(torch.int16), tail, h["n"], h["n_esc"])
    return encoded, scale


def compress(api, raw, metadata):
    torch = api.runtime()
    _, codec = make_codec(api, metadata["dtype"])
    n = int(metadata.get("data_bytes", len(raw)))
    if not 0 < n <= len(raw):
        raise ValueError("Invalid FP8 data size")
    x = torch.frombuffer(bytearray(raw[:n]), dtype=torch.uint8).to("cuda:0")
    codec.calibrate(x)
    return frame(api, codec, codec.encode(x), raw[n:])


def decompress(api, data, expected_bytes, metadata):
    torch = api.runtime()
    module, codec = make_codec(api, metadata["dtype"])
    h, parts = parse(module, codec, data, expected_bytes)
    codec.codebook, codec.window, codec.base_vec = h["codebook"], False, 0
    codec.dec_lut = torch.zeros(256, dtype=torch.uint8, device="cuda:0")
    codec.dec_lut[:codec.K] = torch.tensor(codec.codebook, dtype=torch.uint8, device="cuda:0")
    host, dev, offsets = upload(parts)
    encoded, scale = device_encoded(torch, module, codec, h,
                                    [dev[o:o+len(p)] for o,p in zip(offsets,parts)])
    return api.tensor_bytes(codec.decode(encoded)) + parts[-1]


class Backend(CPUBackend):
    def __init__(self, route):
        super().__init__(route)
        self.api = module_at(route.folder/"api.py")
        self.torch = self.api.runtime()
        self.stream = self.torch.cuda.Stream()
        self.module, self.codec = make_codec(self.api, route.dtype)
        self.data_sizes = None

    def bind_pages(self, pages):
        if pages.layout != "kv" or pages.page_format != "data+fp32-scale-v1":
            raise ValueError("SplitZip FP8 benchmark requires scaled KV pages")
        sizes = {}
        for i in range(len(pages)):
            page = pages.descriptor(i)
            raw, data = page["raw_bytes"], page["data_bytes"]
            if sizes.setdefault(raw, data) != data:
                raise ValueError("Ambiguous FP8 data and scale boundary")
        self.data_sizes = sizes

    def prepare_encode(self, raw):
        n = self.data_sizes[len(raw)] if self.data_sizes is not None else len(raw)
        with self.torch.cuda.stream(self.stream):
            tensor = self.torch.frombuffer(bytearray(raw[:n]), dtype=self.torch.uint8).to("cuda:0")
        return tensor, raw[n:]

    def setup(self, sample):
        tensor, _ = self.prepare_encode(sample)
        with self.torch.cuda.stream(self.stream):
            self.codec.calibrate(tensor)

    def encode(self, prepared):
        tensor, scale = prepared
        return self.codec.encode(tensor), scale

    def finish_encode(self, value, size):
        encoded, scale = value
        return frame(self.api, self.codec, encoded, scale)

    def prepare_decode(self, frames, sizes):
        packets = [parse(self.module, self.codec, f, n) for f,n in zip(frames,sizes)]
        buffers = [part for _,parts in packets for part in parts]
        host, dev, offsets = upload(buffers)
        views = [dev[o:o+len(p)] for o,p in zip(offsets,buffers)]
        items = [device_encoded(self.torch,self.module,self.codec,h,views[i*7:(i+1)*7])
                 for i,(h,_) in enumerate(packets)]
        return dict(call=lambda: [self.codec.decode(encoded) for encoded,_ in items],
                    finish=lambda decoded: [self.torch.cat((raw,scale))
                                             for raw,(_,scale) in zip(decoded,items)],
                    keep=(host,dev,items))

    def decode_signature(self, sizes):
        return tuple(sizes)

    def validate_decoded(self, parts, prepared):
        pass
