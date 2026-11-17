"""CPU decoder for the declared wire format."""
def decode(payload,size,metadata):
    import snappy
    return snappy.decompress(payload)
