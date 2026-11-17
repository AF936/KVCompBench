"""CPU decoder for the declared wire format."""
def decode(payload,size,metadata):
    import zlib
    return zlib.decompress(payload,-15)
