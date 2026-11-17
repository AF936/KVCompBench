"""Install upstream paper codecs in an isolated, project-local Linux CUDA13 environment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import venv

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PLUGINS=ROOT/'benchmark/compressors'
ENV=ROOT/'.venvs/linux-x86_64-paper-cuda13'
PYTHON=ENV/'bin/python'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def download(item,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():
        print('Downloading '+target.name,flush=True)
        with urllib.request.urlopen(urllib.request.Request(item['url'],headers={'User-Agent':'KVCompression/1'}),timeout=180) as source,target.open('xb') as output:
            shutil.copyfileobj(source,output)
    if sha(target)!=item['sha256']:raise ValueError('Download integrity failure: '+str(target))


def extract(archive,target):
    target.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive) as tar:
        for member in tar:
            parts=Path(member.name).parts[1:]
            if not parts or not member.isfile():continue
            path=target/Path(*parts)
            if not path.resolve().is_relative_to(target.resolve()):raise ValueError('Unsafe source archive')
            if path.exists():continue
            path.parent.mkdir(parents=True,exist_ok=True)
            # Set permissions on creation, without changing existing server permissions.
            fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,member.mode&0o777)
            with os.fdopen(fd,'wb') as output:shutil.copyfileobj(tar.extractfile(member),output)


def run(args,**kwargs):
    print('+ '+' '.join(map(str,args)),flush=True)
    subprocess.run(list(map(str,args)),check=True,cwd=ROOT,**kwargs)


def verify_sources():
    for name in ('fse','streamvbyte','alp','chimp','elf','splitzip'):
        folder=PLUGINS/name
        expected=json.loads((folder/'source_hashes.json').read_text())
        for relative,digest in expected.items():
            path=(folder/relative).resolve()
            if not path.is_relative_to(folder.resolve()) or sha(path)!=digest:raise ValueError('Source mismatch: '+str(path))


def native():
    verify_sources()
    build=ENV/'native';build.mkdir(parents=True,exist_ok=True)
    src=PLUGINS/'fse/upstream/lib';out=build/'fse';out.mkdir(exist_ok=True)
    target=out/'libcodec.so'
    if not target.exists():
        run(['gcc','-O2','-fPIC','-shared','-I'+str(src),*[src/(n+'.c') for n in ['fse_compress','fse_decompress','entropy_common','hist','huf_compress','huf_decompress']],'-o',target])
    src=PLUGINS/'streamvbyte/upstream';out=build/'streamvbyte';out.mkdir(exist_ok=True)
    if not (out/'libcodec.so').exists():
        run(['gcc','-O2','-fPIC','-shared','-I'+str(src/'include'),src/'src/streamvbyte_encode.c',src/'src/streamvbyte_decode.c','-o',out/'libcodec.so'])
    src=PLUGINS/'alp/upstream';out=build/'alp';out.mkdir(exist_ok=True)
    if not (out/'libcodec.so').exists():
        # Compile scalar upstream interfaces; do not use CMake CPU probing or fast-math.
        files=['fastlanes_ffor.cpp','fastlanes_unffor.cpp','fastlanes_generated_ffor.cpp','fastlanes_generated_unffor.cpp']
        objects=[]
        for name in files:
            target=out/(name+'.o')
            run(['g++','-std=c++17','-O1','-fno-strict-aliasing','-fPIC','-I'+str(src/'include'),'-c',src/'src'/name,'-o',target])
            objects.append(target)
        run(['g++','-std=c++17','-O1','-fno-strict-aliasing','-fPIC','-shared','-I'+str(src/'include'),PLUGINS/'alp/bridge.cpp',*objects,'-o',out/'libcodec.so'])
    jdk=ENV/'toolchains/jdk17'
    if not (jdk/'bin/javac').is_file():
        item=json.loads((HERE/'install_resources/jdk17.json').read_text());archive=ENV/'downloads/jdk17.tar.gz'
        download(item,archive);extract(archive,jdk)
    dependencies=json.loads((HERE/'install_resources/java-deps.json').read_text())
    for item in dependencies:download(item,build/'java-deps'/item['name'])
    classpath=os.pathsep.join(str(build/'java-deps'/item['name']) for item in dependencies)
    for name in ('chimp','elf'):
        out=build/name/'classes';out.mkdir(parents=True,exist_ok=True)
        if not (out/'source.sha256').exists() or (out/'source.sha256').read_text()!=sha(PLUGINS/name/'source_hashes.json'):
            run([jdk/'bin/javac','-cp',classpath,'-d',out,PLUGINS/name/'JavaCodecBridge.java',*sorted((PLUGINS/name/'upstream/java').rglob('*.java'))])
            (out/'source.sha256').write_text(sha(PLUGINS/name/'source_hashes.json'))
    print('Native upstream codecs ready',flush=True)


def zipnn():
    # Fixed upstream 0.5.4 source with the already documented CPython method-table terminator fix.
    folder=PLUGINS/'zipnn';item=json.loads((folder/'upstream.json').read_text());archive=folder/'upstream-0.5.4.tar.gz'
    if sha(archive)!=item['sha256']:raise ValueError('ZipNN source integrity failure')
    source=ENV/'build/zipnn-0.5.4-sentinel1'
    if not source.exists():extract(archive,source)
    path=source/'csrc/zipnn_core_module.c';text=path.read_text();marker='"Combine four buffers into a single bytearray using dtype16"},\n};'
    if marker in text:path.write_text(text.replace(marker,marker[:-2]+'    {NULL, NULL, 0, NULL}\n};'))
    elif '{NULL, NULL, 0, NULL}' not in text:raise ValueError('Unexpected ZipNN method table')
    path=source/'setup.py';text=path.read_text();path.write_text(text.replace('version="0.5.4"','version="0.5.4+sentinel1"'))
    run([PYTHON,'-m','pip','install','--no-deps','--no-build-isolation','--no-compile',source])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--native-only',action='store_true');parser.add_argument('--index-url',default='https://pypi.org/simple');args=parser.parse_args()
    if sys.platform!='linux' or platform.machine().lower() not in ('x86_64','amd64'):raise RuntimeError('This build profile currently requires Linux x86_64')
    if ROOT.resolve().is_relative_to(Path('/data')) and not ROOT.resolve().is_relative_to(Path('/data/lxy')):raise RuntimeError('Shared-server path is outside /data/lxy')
    scratch=ENV/'tmp';scratch.mkdir(parents=True,exist_ok=True)
    os.environ.update(TMPDIR=str(scratch),TEMP=str(scratch),TMP=str(scratch),PIP_CACHE_DIR=str(ENV/'pip-cache'),PYTHONDONTWRITEBYTECODE='1',MAX_JOBS='1')
    if not PYTHON.exists():venv.EnvBuilder(with_pip=True,symlinks=False).create(ENV)
    if args.native_only:native();return
    wheelhouse=ENV/'wheelhouse'
    if not wheelhouse.is_dir():run([PYTHON,'-m','pip','install','--upgrade','pip','--index-url',args.index_url])
    probe=subprocess.run([str(PYTHON),'-c',"import importlib.metadata as m; assert m.version('torch').startswith('2.9.1+cu130')"],capture_output=True)
    if probe.returncode:run([PYTHON,'-m','pip','install','--no-compile','torch==2.9.1','--index-url','https://download.pytorch.org/whl/cu130'])
    source=['--no-index','--find-links',str(wheelhouse)] if wheelhouse.is_dir() else ['--index-url',args.index_url]
    run([PYTHON,'-m','pip','install','--no-compile','-r',HERE/'requirements-paper.txt',*source])
    native();zipnn()
    run([PYTHON,'-m','pip','check'])
    frozen=subprocess.check_output([str(PYTHON),'-m','pip','freeze'],text=True)
    (ENV/'requirements.lock.txt').write_text(frozen)
    (ENV/'environment.json').write_text(json.dumps(dict(name=ENV.name,profile='paper-cuda13',python=str(PYTHON)),indent=2))
    from setup_environment import activate_setting
    activate_setting(ENV.name)
    print('Selected environment: '+ENV.name,flush=True)
    print('GPU tests require an explicitly assigned CUDA_VISIBLE_DEVICES; installation does not run them.')


if __name__=='__main__':main()
