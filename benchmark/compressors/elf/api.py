"""Invoke the pinned upstream Java codec through a counted, bit-exact byte bridge."""
import os
from pathlib import Path
import shutil
import subprocess
import sys


def call(data,metadata,mode):
    name=Path(__file__).parent.name
    from codec_manager import artifact_path
    classes=artifact_path(name,'native/'+name+'/classes/JavaCodecBridge.class').parent
    bundled=artifact_path(name,'toolchains/jdk17/bin/java')
    java=str(bundled) if bundled.is_file() else shutil.which('java')
    if not java or not (classes/'JavaCodecBridge.class').is_file():raise ImportError('Build Java codec dependencies with python3 kvbench.py setup --codecs elf')
    width={'fp32':4,'fp64':8}[metadata['dtype']]
    classpath=os.pathsep.join([str(classes),str(artifact_path(name,'native/java-deps/dsiutils-2.7.0.jar')),str(artifact_path(name,'native/java-deps/fastutil-8.4.4.jar'))])
    scratch=Path(sys.prefix)/'tmp';scratch.mkdir(parents=True,exist_ok=True)
    result=subprocess.run([java,'-Xmx256m','-Djava.io.tmpdir='+str(scratch),'-cp',classpath,'JavaCodecBridge',mode,name,str(width)],input=data,capture_output=True,timeout=60)
    if result.returncode:raise RuntimeError(name+' upstream Java bridge failed: '+result.stderr.decode(errors='replace')[:2048])
    return result.stdout


def compress(raw,metadata):return call(raw,metadata,'encode')


def decompress(data,expected_bytes,metadata):
    raw=call(data,metadata,'decode')
    if len(raw)!=expected_bytes:raise ValueError('Java codec length mismatch')
    return raw
