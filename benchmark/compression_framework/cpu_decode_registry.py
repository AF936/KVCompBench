"""Decoder declarations are codec-local; shared formats use a common registry."""
import json,importlib.util
from pathlib import Path

def resolve_decoder(route):
 if route.mode.endswith('-CPU'):return None,'Original CPU decoder'
 p=route._folder/'decode_backends.json'
 if not p.exists():return None,'No declared compatible CPU decoder for the existing frame format'
 entry=json.loads(p.read_text()).get(route.name,{}).get('cpu')
 if not entry:return None,'No declared compatible CPU decoder'
 if entry['type']=='shared':
  from gpu_decode import unpack_frame
  ident=entry['id']
  def decode(frame,size):
   backend,n,payload=unpack_frame(frame)
   if n!=size or backend!=ident:raise ValueError('CPU decoder frame mismatch')
   import gzip,zlib,lz4.block,snappy,zstandard
   functions={'nvcomp-gzip-raw':gzip.decompress,'nvcomp-deflate-raw':lambda x:zlib.decompress(x,-15),'nvcomp-lz4-block-raw':lambda x:lz4.block.decompress(x,uncompressed_size=size),'nvcomp-snappy-raw':snappy.decompress,'nvcomp-zstd-raw':lambda x:zstandard.ZstdDecompressor().decompress(x,max_output_size=size)}
   return functions[ident](payload)
  return decode,'shared:'+ident
 if entry['type']=='module':
  base=route._folder.resolve();path=(base/entry['file']).resolve();assert path.is_relative_to(base)
  spec=importlib.util.spec_from_file_location('cpu_decoder_'+route.name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  function=getattr(module,entry['function'])
  if entry.get('unpack_frame'):
   from gpu_decode import unpack_frame
   return lambda frame,n:function(unpack_frame(frame)[2],n,{'dtype':'bf16'}),'module:'+entry['file']+':'+entry['function']
  return function,'module:'+entry['file']+':'+entry['function']
 raise ValueError('Unknown CPU decoder declaration')
