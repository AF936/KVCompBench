"""CPU decoder for the declared wire format."""
def decode(payload,size,metadata):
    import lz4.block
    return lz4.block.decompress(payload,uncompressed_size=size)
