"""Official zfpy default mode is reversible; no tolerance, precision or rate override."""
def compress(raw,metadata):
    import numpy as np
    import zfpy
    dtype={'fp32':'<f4','fp64':'<f8'}[metadata['dtype']]
    array=np.frombuffer(raw,dtype=dtype).copy()
    return zfpy.compress_numpy(array,write_header=True)


def decompress(data,expected_bytes,metadata):
    import numpy as np
    import zfpy
    array=zfpy.decompress_numpy(data)
    dtype=np.dtype({'fp32':'<f4','fp64':'<f8'}[metadata['dtype']])
    if array.dtype!=dtype or array.nbytes!=expected_bytes:raise ValueError('Unexpected ZFP dtype or length')
    return array.tobytes()
