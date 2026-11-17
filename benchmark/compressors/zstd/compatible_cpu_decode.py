"""CPU decoder for the declared wire format."""
def decode(payload,size,metadata):
    import zstandard
    return zstandard.ZstdDecompressor().decompress(payload,max_output_size=size)
