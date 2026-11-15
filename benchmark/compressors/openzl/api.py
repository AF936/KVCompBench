"""Official OpenZL serial-byte graph; compressed frames carry decoding graphs."""
from functools import lru_cache

@lru_cache(maxsize=1)
def compressor():
    import openzl.ext as zl
    obj=zl.Compressor();obj.select_starting_graph(zl.graphs.Compress()(obj))
    return obj


def compress(raw,metadata):
    import openzl.ext as zl
    context=zl.CCtx();context.ref_compressor(compressor());context.set_parameter(zl.CParam.FormatVersion,zl.MAX_FORMAT_VERSION)
    return context.compress([zl.Input(zl.Type.Serial,raw)])


def decompress(data,expected_bytes,metadata):
    import openzl.ext as zl
    outputs=zl.DCtx().decompress(data)
    if len(outputs)!=1 or outputs[0].type!=zl.Type.Serial:raise ValueError('Expected one serial OpenZL output')
    return outputs[0].content.as_bytes()
