"""Tensor-aware BF16 and scaled FP8 pages with byte-exact payload layouts."""
import configparser
import itertools
import math
from pathlib import Path


class Pages:
    def __init__(self, path, config):
        import torch
        self.torch = torch
        value = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        self.metadata = value if isinstance(value, dict) else {}
        self.tensor = value['data'] if isinstance(value, dict) else value
        names = {torch.bfloat16: 'bf16', torch.float8_e4m3fn: 'fp8_e4m3fn', torch.float8_e5m2: 'fp8_e5m2'}
        if not isinstance(self.tensor, torch.Tensor) or self.tensor.dtype not in names:
            raise ValueError('Expected BF16, FP8 E4M3FN or FP8 E5M2 tensor')
        self.dtype = names[self.tensor.dtype]
        if config.get('dtype', 'auto') not in ('auto', self.dtype):
            raise ValueError('Configured dtype does not match stored tensor')
        self.element_bytes = self.tensor.element_size()
        self.shape = list(self.tensor.shape)
        self.scale = None
        quantization = self.metadata.get('quantization', {})
        if self.dtype.startswith('fp8'):
            self.scale = self.metadata.get('scale')
            if self.scale is None:
                self.scale = quantization.get('scale')
            if not isinstance(self.scale, torch.Tensor):
                raise ValueError('Scaled FP8 input requires an FP32 scale tensor')
            if self.scale.dtype != torch.float32 or list(self.scale.shape) != self.shape[:-1] + [1]:
                raise ValueError('FP8 scale must be FP32 with one value per last-axis vector')
            axis = quantization.get('axis', self.metadata.get('scale_axis', -1))
            if axis not in (-1, len(self.shape)-1, [len(self.shape)-1], [-1]):
                raise ValueError('Unsupported FP8 quantization axis')
            if quantization.get('fp8_format', self.dtype.removeprefix('fp8_')) != self.dtype.removeprefix('fp8_'):
                raise ValueError('Quantization format does not match stored FP8 dtype')
        self.page_format = 'data+fp32-scale-v1' if self.scale is not None else 'tensor-bytes-v1'
        keys = ('format', 'conversion', 'key_value_shared', 'prequant_bf16_exactly_recoverable', 'source_fp8_dtype',
                'source_file', 'source_format', 'source_dtype', 'pre_quantization_dtype', 'stored_dtype', 'scheme',
                'shape_semantics', 'quantization_metrics')
        self.dataset_metadata = {k: self.metadata[k] for k in keys if k in self.metadata}
        source_dtype = self.metadata.get('pre_quantization_dtype', self.metadata.get('source_dtype'))
        self.quantization_origin = 'requantized_bf16' if self.scale is not None and source_dtype == 'torch.bfloat16' else 'unspecified'
        self.dataset_metadata.update(storage_dtype=self.dtype, page_format=self.page_format, quantization_origin=self.quantization_origin)
        if self.scale is not None:
            self.dataset_metadata['quantization'] = {k: v for k, v in quantization.items() if k != 'scale'}
            self.dataset_metadata.update(scale_shape=list(self.scale.shape), scale_dtype='fp32',
                                         scale_axis=self.metadata.get('scale_axis', len(self.shape)-1))
        self.config = dict(config)
        self.layout = config['layout']
        if self.layout == 'flat':
            self.block_bytes = int(config['block_bytes'])
            if self.block_bytes <= 0 or self.block_bytes % self.element_bytes:
                raise ValueError('Flat block size must align to the tensor element size')
            data = self.tensor.contiguous().reshape(-1).view(torch.uint8)
            self.flat_data_bytes = data.numel()
            if self.scale is not None:
                data = torch.cat((data, self.scale.contiguous().reshape(-1).view(torch.uint8)))
            self.flat = data.numpy()
            self.count = math.ceil(len(self.flat)/self.block_bytes)
        else:
            self.axes = [x.strip() for x in config['axes'].split(',')]
            if len(self.axes) != len(self.shape) or len(set(self.axes)) != len(self.axes) or not {'layer', 'token'} <= set(self.axes):
                raise ValueError('Explicit unique layer/token axes required')
            if any(x not in {'layer', 'kv', 'batch', 'head', 'token', 'dim', 'latent'} for x in self.axes):
                raise ValueError('Unsupported axis name')
            if config.get('kv_semantics') == 'shared' and 'kv' in self.axes:
                raise ValueError('Shared latent cannot have a fabricated K/V axis')
            if self.scale is not None and self.axes[-1] not in ('dim', 'latent'):
                raise ValueError('Scaled FP8 requires dim or latent as the last axis')
            self.token_axis = self.axes.index('token')
            self.tokens = int(config['page_tokens'])
            if self.tokens <= 0:
                raise ValueError('page_tokens must be positive')
            self.outer_axes = [i for i, x in enumerate(self.axes) if x in ('layer', 'batch')]
            self.outer = list(itertools.product(*(range(self.shape[i]) for i in self.outer_axes)))
            self.per_outer = math.ceil(self.shape[self.token_axis]/self.tokens)
            self.count = len(self.outer)*self.per_outer
            elements = self.tokens*math.prod(n for i, n in enumerate(self.shape) if i not in self.outer_axes+[self.token_axis])
            self.block_bytes = elements*self.element_bytes
            if self.scale is not None:
                self.block_bytes += elements//self.shape[-1]*4

    def __len__(self):
        return self.count

    def descriptor(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self.layout == 'flat':
            start = index*self.block_bytes
            size = min(self.block_bytes, len(self.flat)-start)
            data_bytes = max(0, min(size, self.flat_data_bytes-start))
            result = dict(index=index, byte_start=start, layout='flat')
        else:
            outer, page = divmod(index, self.per_outer)
            start = page*self.tokens
            stop = min(start+self.tokens, self.shape[self.token_axis])
            elements = (stop-start)*math.prod(n for i, n in enumerate(self.shape) if i not in self.outer_axes+[self.token_axis])
            data_bytes = elements*self.element_bytes
            size = data_bytes+(elements//self.shape[-1]*4 if self.scale is not None else 0)
            result = dict(index=index, layout='kv', coordinates={self.axes[i]: v for i, v in zip(self.outer_axes, self.outer[outer])},
                          token_start=start, token_stop=stop, axes=self.axes, kv_semantics=self.config.get('kv_semantics', 'separate'))
        result.update(storage_dtype=self.dtype, page_format=self.page_format, data_bytes=data_bytes,
                      scale_offset=data_bytes, scale_bytes=size-data_bytes, raw_bytes=size)
        return result

    def __getitem__(self, index):
        descriptor = self.descriptor(index)
        if self.layout == 'flat':
            return self.flat[index*self.block_bytes:(index+1)*self.block_bytes].tobytes()
        slices = [slice(None)]*len(self.shape)
        for i, v in zip(self.outer_axes, self.outer[index//self.per_outer]):
            slices[i] = slice(v, v+1)
        slices[self.token_axis] = slice(descriptor['token_start'], descriptor['token_stop'])
        data = self.tensor[tuple(slices)].contiguous().view(self.torch.uint8).numpy().tobytes()
        if self.scale is not None:
            data += self.scale[tuple(slices)].contiguous().view(self.torch.uint8).numpy().tobytes()
        return data

    def split_page(self, index, raw):
        descriptor = self.descriptor(index)
        if len(raw) != descriptor['raw_bytes']:
            raise ValueError('Decoded page size mismatch')
        boundary = descriptor['scale_offset']
        return raw[:boundary], raw[boundary:]


def dataset_configs(settings):
    c = configparser.ConfigParser(interpolation=None)
    c.read(settings, encoding='utf-8')
    result = {}
    for filename, block in c['benchmark_datasets'].items():
        if Path(filename).name != filename:
            raise ValueError('Dataset must be a filename')
        section = 'dataset:'+filename
        values = dict(c[section]) if c.has_section(section) else dict(layout='flat')
        if values.get('layout') not in ('kv', 'flat'):
            raise ValueError('layout must be kv or flat')
        values['block_bytes'] = int(block)
        result[filename] = values
    return result


class PageSubset:
    def __init__(self, base, indices):
        self.base = base
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getattr__(self, key):
        return getattr(self.base, key)

    def __getitem__(self, index):
        return self.base[self.indices[index]]

    def descriptor(self, index):
        return dict(self.base.descriptor(self.indices[index]), validation_source_index=self.indices[index])


def validation_subset(pages):
    return PageSubset(pages, sorted(set([0, min(1, len(pages)-1), pages.per_outer-1 if pages.layout == 'kv' else len(pages)-1])))
