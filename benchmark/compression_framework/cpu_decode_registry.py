"""Decoder declarations are codec-local; shared formats use a common registry."""
import json,importlib.util
from pathlib import Path

def resolve_decoder(route):
 if route.mode.endswith('-CPU'):return None,'Original CPU decoder'
 p=route._folder/'decode_backends.json'
 if not p.exists():return None,'No declared compatible CPU decoder for the existing frame format'
 entry=json.loads(p.read_text()).get(route.name,{}).get('cpu')
 if not entry:return None,'No declared compatible CPU decoder'
 if entry['type']=='module':
  base=route._folder.resolve();path=(base/entry['file']).resolve();assert path.is_relative_to(base)
  spec=importlib.util.spec_from_file_location('cpu_decoder_'+route.name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  function=getattr(module,entry['function'])
  if entry.get('unpack_frame'):
   from gpu_decode import unpack_frame
   def decode(frame,n):
    identifier,size,payload=unpack_frame(frame)
    expected=route.backend_id.removeprefix('KVG1:')
    if identifier!=expected or size!=n:raise ValueError('CPU decoder frame mismatch')
    return function(payload,n,{'dtype':route.dtype})
   return decode,'module:'+entry['file']+':'+entry['function']
  return function,'module:'+entry['file']+':'+entry['function']
 raise ValueError('Unknown CPU decoder declaration')
