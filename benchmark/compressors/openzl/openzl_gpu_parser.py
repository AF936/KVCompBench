"""Strict OpenZL v27 single-output, single-Zstd-node frame parser draft."""
import struct


def varint(data, pos):
    result = 0
    for shift in range(0, 64, 7):
        if pos >= len(data):
            raise ValueError('Truncated OpenZL integer')
        byte = data[pos]
        pos += 1
        result |= (byte & 127) << shift
        if byte < 128:
            return result, pos
    raise ValueError('Invalid OpenZL integer')


def parse(data, expected_bytes, hash64=None):
    if len(data) < 10 or struct.unpack_from('<I', data)[0] != 0xD7B1A5C0 + 27:
        raise ValueError('Expected OpenZL format 27')
    flags = data[4]
    if flags & ~3 or data[5] != 1:
        raise ValueError('Expected single Serial output without comments/dictionaries')
    declared, pos = varint(data, 6)
    if declared != expected_bytes + 1:
        raise ValueError('OpenZL output size mismatch')
    header_checksum = None
    if flags & 2:
        if hash64 is None:
            raise ValueError('OpenZL checksum implementation required')
        header_checksum = data[pos]
        if hash64(data[:pos]) & 255 != header_checksum:
            raise ValueError('OpenZL header checksum mismatch')
        pos += 1
    pieces = []
    while True:
        chunk_start = pos
        count, pos = varint(data, pos)
        if count == 0:
            break
        stored, pos = varint(data, pos)
        if count not in (1, 2) or stored != 1:
            raise ValueError('Expected single decoder and stored stream')
        # Standard node 22 (Zstd), no private header, no variable outputs,
        # one regeneration, no dictionary, output stream distance zero.
        if count == 2 and data[pos:pos + 6] != bytes([0, 22, 0, 0, 0, 0]):
            raise ValueError('Unsupported OpenZL decoding graph')
        if count == 2:
            pos += 6
        size, pos = varint(data, pos)
        if pos + size > len(data):
            raise ValueError('Truncated OpenZL stored stream')
        payload = data[pos:pos + size]
        pos += size
        checks = []
        for enabled in (flags & 1, flags & 2):
            if enabled:
                if pos + 4 > len(data):
                    raise ValueError('Truncated OpenZL checksum')
                checks.append(struct.unpack_from('<I', data, pos)[0])
                pos += 4
            else:
                checks.append(None)
        if flags & 2 and (hash64(data[chunk_start:pos - 4]) & 0xFFFFFFFF) != checks[1]:
            raise ValueError('OpenZL compressed checksum mismatch')
        pieces.append((payload, checks, count == 1))
    if pos != len(data):
        raise ValueError('Trailing OpenZL bytes')
    return flags, header_checksum, pieces
