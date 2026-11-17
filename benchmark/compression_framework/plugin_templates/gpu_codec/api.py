"""Replace this byte-preserving example with the CPU codec API."""

def compress(raw, metadata):
    return bytes(raw)


def decompress(packet, expected_bytes, metadata):
    if len(packet) != expected_bytes:
        raise ValueError("Decoded length mismatch")
    return bytes(packet)
