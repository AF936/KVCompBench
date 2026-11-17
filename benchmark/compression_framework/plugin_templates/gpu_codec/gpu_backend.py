"""Codec-local GPU backend template.

The framework wraps the payload from cpu_encode or gpu_encode in a KVG1 frame.
gpu_decode receives that payload and the decoded byte count, and must return
the exact original bytes.
"""

def cpu_encode(raw):
    return bytes(raw)


def gpu_encode(raw):
    import torch
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8).to("cuda:0").cpu().numpy().tobytes()


def gpu_decode(payload, expected_bytes):
    if len(payload) != expected_bytes:
        raise ValueError("Payload length mismatch")
    import torch
    return torch.frombuffer(bytearray(payload), dtype=torch.uint8).to("cuda:0").cpu().numpy().tobytes()
