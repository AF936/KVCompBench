"""Discover trusted codec packages from compressors/<package>/codec.json."""
from __future__ import annotations
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

DEFAULT_COMPRESSORS = Path(__file__).resolve().parents[1] / 'compressors'


def contained(root, value):
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'Plugin path escapes package: {value}')
    return path


def manifests(directory):
    root = Path(directory).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'Compressor directory not found: {root}')
    for folder in sorted(root.iterdir()):
        if folder.name.startswith(('.', '_')) or not folder.is_dir():
            continue
        path = folder / 'codec.json'
        if not path.exists():
            continue
        contained(root, str(folder.relative_to(root)))
        contained(folder, 'codec.json')
        try:
            value = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(value, dict) or type(value.get('schema_version')) is not int or value.get('schema_version') != 1:
                raise ValueError('schema_version must be 1')
            if set(value) - {'schema_version','codecs'} or not isinstance(value.get('codecs'),list):
                raise ValueError('Expected schema_version and codecs array')
            for entry in value['codecs']:
                if not isinstance(entry,dict):
                    raise ValueError('Each codec must be an object')
                yield folder, entry
        except (ValueError,TypeError) as error:
            raise ValueError(f'{path}: {error}') from error


class PackageCodec:
    def __init__(self, folder, spec):
        self.folder, self.spec = folder.resolve(), dict(spec)
        self.NAME = spec['name']
        allowed = {'name','kind','version','device','dtypes','alignment','enabled','description',
                   'module','file','compress','decompress','interface','compress_kwargs',
                   'decompress_kwargs','distribution','timeout_seconds',
                   'gpu_decode_backend','gpu_decode_frame_version','gpu_backend'}
        if set(spec) - allowed:
            raise ValueError(f'{self.NAME}: unknown manifest fields {set(spec)-allowed}')
        if not isinstance(spec.get('version'),str) or not spec['version']:
            raise ValueError(f'{self.NAME}: version must be a nonempty string')
        if spec.get('device','cpu') not in ('cpu','gpu'):
            raise ValueError('device must be cpu or gpu')
        if type(spec.get('alignment',1)) is not int or spec.get('alignment',1) < 1:
            raise ValueError('alignment must be a positive integer')
        if 'dtypes' in spec and (not isinstance(spec['dtypes'],list) or not all(isinstance(x,str) for x in spec['dtypes'])):
            raise ValueError('dtypes must be an array of strings')
        self.kind = spec['kind']
        self.gpu_decode = spec.get('gpu_backend')
        if self.gpu_decode is not None and not isinstance(self.gpu_decode, dict):
            raise ValueError('gpu_backend must be an object')
        if 'gpu_decode_backend' in spec:
            backend = spec['gpu_decode_backend']
            frame_version = spec.get('gpu_decode_frame_version', 1)
            if not isinstance(backend, str) or not re.fullmatch(r'[a-z0-9][a-z0-9-]*', backend):
                raise ValueError('gpu_decode_backend must be a lowercase backend ID')
            if type(frame_version) is not int or frame_version != 1:
                raise ValueError('gpu_decode_frame_version must be 1')
            self.gpu_decode = dict(backend=backend, frame_version=frame_version)
        elif 'gpu_decode_frame_version' in spec:
            raise ValueError('gpu_decode_frame_version requires gpu_decode_backend')
        if self.kind == 'python':
            if ('module' in spec) == ('file' in spec):
                raise ValueError('Specify exactly one of module or file')
            if spec.get('interface','bytes') not in ('bytes','context'):
                raise ValueError('interface must be bytes or context')
            for key in ('compress','decompress'):
                if not isinstance(spec.get(key),str) or not spec[key]:
                    raise ValueError(f'{key} must identify a callable')
            for key in ('compress_kwargs','decompress_kwargs'):
                if not isinstance(spec.get(key,{}),dict):
                    raise ValueError(f'{key} must be an object')
            if 'file' in spec and not contained(self.folder,spec['file']).is_file():
                raise FileNotFoundError(spec['file'])
        elif self.kind == 'command':
            for key in ('compress','decompress'):
                command = spec.get(key)
                if not isinstance(command,list) or not command or not all(isinstance(x,str) and x for x in command):
                    raise ValueError(f'{key} must be a nonempty argument array')
                if not any('{input}' in x for x in command) or not any('{output}' in x for x in command):
                    raise ValueError(f'{key} requires input and output placeholders')
                for part in command:
                    # Reject unknown placeholders early; no shell is involved.
                    part.format_map(dict(input='in',output='out',expected_bytes=1,plugin_dir='package',python='python'))
            timeout = spec.get('timeout_seconds',60)
            if type(timeout) not in (int,float) or not 0 < timeout <= 3600:
                raise ValueError('timeout_seconds must be in (0, 3600]')
        else:
            raise ValueError('kind must be python, command, or builtin')
        self.entry = dict(adapter='gpu-api' if spec.get('device')=='gpu' else 'package-api',
                          provenance=f'Auto-discovered {self.kind} package: {folder.name}')
        identity = json.dumps(spec,sort_keys=True,allow_nan=False).encode()
        if 'file' in spec:
            identity += contained(self.folder,spec['file']).read_bytes()
        self.identity = hashlib.sha256(identity).hexdigest()[:16]
        self.VERSION = f'package-v1/{spec["version"]}/{self.identity}'
        self._functions = None

    def supports(self, metadata):
        return 'dtypes' not in self.spec or metadata.get('dtype','bytes') in self.spec['dtypes']

    def prepare(self, metadata):
        if not self.supports(metadata):
            raise ValueError(f'{self.NAME}: unsupported dtype')
        if self.spec.get('device') == 'gpu' and not os.environ.get('CUDA_VISIBLE_DEVICES','').strip():
            raise RuntimeError('Set CUDA_VISIBLE_DEVICES to an administrator-assigned GPU')
        if self.kind == 'python' and self._functions is None:
            if 'module' in self.spec:
                module = importlib.import_module(self.spec['module'])
            else:
                path = contained(self.folder,self.spec['file'])
                # Relative imports work for a package whose entry is __init__.py.
                unique = 'kv_package_' + hashlib.sha256(str(self.folder).encode()).hexdigest()[:12] + '_' + self.identity
                module_spec = importlib.util.spec_from_file_location(unique,path,
                    submodule_search_locations=[str(path.parent)] if path.name=='__init__.py' else None)
                if module_spec is None or module_spec.loader is None:
                    raise ValueError(f'Cannot load Python API file: {path}')
                module = importlib.util.module_from_spec(module_spec)
                sys.modules[unique] = module
                try:
                    module_spec.loader.exec_module(module)
                except BaseException:
                    sys.modules.pop(unique,None)
                    raise
            def function(name):
                value = module
                for part in name.split('.'):
                    value = getattr(value,part)
                if not callable(value):raise ValueError(f'{name} is not callable')
                return value
            self._functions = (function(self.spec['compress']),function(self.spec['decompress']))
        suffix = ''
        if 'distribution' in self.spec:
            suffix = '/' + importlib.metadata.version(self.spec['distribution'])
        self.VERSION = f'package-v1/{self.spec["version"]}/{self.identity}{suffix}'

    def _command(self, data, mode, expected_bytes):
        # Scratch files stay inside the project, including on the shared server.
        scratch = Path(__file__).resolve().parent / '.runtime' / 'plugin-tmp'
        scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='codec-',dir=scratch) as directory:
            root = Path(directory)
            source, target = root/'input.bin', root/'output.bin'
            source.write_bytes(data)
            values = dict(input=str(source),output=str(target),expected_bytes=expected_bytes,
                          plugin_dir=str(self.folder),python=sys.executable)
            arguments = [part.format_map(values) for part in self.spec[mode]]
            if arguments[0].startswith('./'):
                arguments[0] = str(contained(self.folder,arguments[0]))
            result = subprocess.run(arguments,cwd=self.folder,stdin=subprocess.DEVNULL,
                capture_output=True,timeout=self.spec.get('timeout_seconds',60),check=False)
            if result.returncode:
                raise RuntimeError(f'{self.NAME} {mode} exited {result.returncode}: '+result.stderr.decode(errors='replace')[:2048])
            if not target.is_file():raise ValueError(f'{self.NAME}: no output file was produced')
            if mode=='decompress' and target.stat().st_size != expected_bytes:
                raise ValueError(f'{self.NAME}: decoded length mismatch')
            return target.read_bytes()

    def compress(self, raw, metadata):
        self.prepare(metadata)
        if len(raw) % self.spec.get('alignment',1):
            raise ValueError(f'{self.NAME}: block length violates alignment')
        if self.kind=='command':return self._command(raw,'compress',len(raw))
        args = (raw,dict(metadata)) if self.spec.get('interface','bytes')=='context' else (raw,)
        result = self._functions[0](*args,**self.spec.get('compress_kwargs',{}))
        if not isinstance(result,(bytes,bytearray,memoryview)):
            raise TypeError('compress must return a bytes-like value')
        return bytes(result)

    def decompress(self, packet, expected_bytes, metadata):
        self.prepare(metadata)
        if expected_bytes % self.spec.get('alignment',1):raise ValueError('Invalid output alignment')
        if self.kind=='command':return self._command(packet,'decompress',expected_bytes)
        args = (packet,expected_bytes,dict(metadata)) if self.spec.get('interface','bytes')=='context' else (packet,)
        result = self._functions[1](*args,**self.spec.get('decompress_kwargs',{}))
        if not isinstance(result,(bytes,bytearray,memoryview)):
            raise TypeError('decompress must return a bytes-like value')
        raw = bytes(result)
        if len(raw)!=expected_bytes:raise ValueError('Decoded length mismatch')
        return raw


def discover(directory=DEFAULT_COMPRESSORS, builtins=None):
    builtins = builtins or {}
    plugins = {}
    for folder, spec in manifests(directory):
        try:
            if type(spec.get('enabled',True)) is not bool:
                raise ValueError('enabled must be a boolean')
            if not spec.get('enabled',True):continue
            name = spec.get('name')
            if not isinstance(name,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.:+-]*',name):
                raise ValueError('Invalid codec name')
            if name in plugins:raise ValueError(f'Duplicate codec name: {name}')
            if spec.get('kind')=='builtin':
                if set(spec)-{'name','kind','codec','enabled','gpu_backend','dtypes'}:raise ValueError('Unknown builtin manifest fields')
                target = spec.get('codec',name)
                if name!=target or target not in builtins:raise ValueError('Unknown or renamed builtin codec')
                plugins[name] = builtins[target]
            else:
                if name in builtins:raise ValueError(f'Codec name conflicts with builtin: {name}')
                plugins[name] = PackageCodec(folder,spec)
        except (ValueError,KeyError,TypeError,FileNotFoundError) as error:
            raise ValueError(f'{folder / "codec.json"}: {error}') from error
    return plugins


def dependency_files(directory=DEFAULT_COMPRESSORS):
    roots = {folder for folder,spec in manifests(directory) if spec.get('enabled',True)}
    return [contained(folder,'requirements.txt') for folder in sorted(roots)
            if (folder/'requirements.txt').is_file()]
