"""CPU decoder for the declared wire format."""
def decode(payload,size,metadata):
    import gzip
    return gzip.decompress(payload)
