"""OpenZL 0.3 Zstd-node payload normalization; outer graph parsing is separate."""


def standard_zstd_payload(payload, format_version):
    value, shift, offset = 0, 0, 0
    while True:
        if offset >= len(payload) or shift >= 64:
            raise ValueError('Truncated OpenZL Zstd element width')
        byte = payload[offset]
        offset += 1
        value |= (byte & 127) << shift
        if byte < 128:
            break
        shift += 7
    if value != 1:
        raise ValueError('Expected OpenZL Serial element width 1')
    frame = payload[offset:]
    if not frame:
        raise ValueError('Missing OpenZL Zstd frame')
    return (b'\x28\xb5\x2f\xfd' + frame) if format_version >= 9 else frame
