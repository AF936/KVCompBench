"""Generic per-project JSONL benchmark; result.csv is output only."""
from pathlib import Path
import sys,os,json,csv,hashlib,time,configparser,importlib.metadata as md,importlib.util
from itertools import islice

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
 return h.hexdigest()

def fingerprint(spec):return hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def load_journal(path):
 """Load committed transactions; quarantine only an incomplete final write."""
 result={}
 if not path.exists():return result
 data=path.read_bytes();lines=data.splitlines(keepends=True);valid=0
 for index,line in enumerate(lines):
  try:batch=json.loads(line)
  except json.JSONDecodeError:
   if index!=len(lines)-1 or line.endswith(b'\n'):raise
   fragment=data[valid:]
   backup=path.with_name(path.name+'.incomplete-'+hashlib.sha256(fragment).hexdigest()[:12]+'.jsonpart')
   if not backup.exists():backup.write_bytes(fragment)
   with path.open('r+b') as handle:handle.truncate(valid)
   break
  for record in batch['records']:result[(batch['job'],record['block'])]=record
  valid+=len(line)
 else:
  if data and not data.endswith(b'\n'):
   with path.open('ab') as handle:handle.write(b'\n')
 return result

def main():
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--validate-only',action='store_true');args=parser.parse_args()
 root=args.root.resolve();out=args.output.resolve();assert out.is_relative_to(root/'benchmark/output');out.mkdir(parents=True,exist_ok=True)
 sys.path[:0]=[str(root/'benchmark'),str(root/'benchmark/compression_framework')]
 from benchmark_engine import encoded_blocks,load_tensor,dataset_blocks,write_png,cpu_threads,decode_batch_blocks
 from benchmark_routes import discover_routes
 from batch_decode import BatchDecoder
 from codec_timing import TIMING_VERSION
 from cpu_decode_registry import resolve_decoder
 routes=discover_routes(root/'benchmark/compressors');folders={c['name']:p.parent for p in (root/'benchmark/compressors').glob('*/codec.json') for c in json.loads(p.read_text())['codecs']};config=root/'benchmark/settings.txt';workers=cpu_threads(config);batch_size=decode_batch_blocks(config)
 common={p.name:digest(p) for p in [Path(__file__),root/'benchmark/benchmark_engine.py',root/'benchmark/compression_framework/codec_timing.py',root/'benchmark/compression_framework/batch_decode.py',root/'benchmark/compression_framework/cpu_decode_registry.py',root/'benchmark/compression_framework/benchmark_routes.py']}
 import torch
 device=torch.cuda.get_device_name(0);properties=torch.cuda.get_device_properties(0)
 environment=dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,device=device,capability=[properties.major,properties.minor],nvcomp=md.version('nvidia-nvcomp-cu13'))
 journal=out/'progress.jsonl';done=load_journal(journal);rows=[];capabilities=[]
 for filename,block_bytes in dataset_blocks(config).items():
  path=root/'benchmark/input'/filename;data=load_tensor(path);input_hash=digest(path)
  if args.validate_only:data=data[:block_bytes*2+128]
  for name,route in routes.items():
   folder=folders[name].resolve();route._folder=folder;assert folder.is_relative_to(root)
   source={str(p.relative_to(folder)):digest(p) for p in sorted(folder.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.json','.so','.dll','.pyd') and p.resolve().is_relative_to(root)}
   cpu,description=resolve_decoder(route)
   modes=[(route.mode,None)]
   if route.mode.endswith('-GPU') and cpu is not None:modes.append((route.mode.split('-')[0]+'-CPU',cpu))
   capabilities.append(dict(codec=name,original=route.mode,cpu_available=cpu is not None or route.mode.endswith('-CPU'),cpu_note=description))
   for mode,cpu_decoder in modes:
    gpu_decode=mode.endswith('-GPU');group_size=batch_size if gpu_decode else 1
    packages={}
    requirements=folder/'runtime_requirements.json'
    declaration=next(c for c in json.loads((folder/'codec.json').read_text())['codecs'] if c['name']==name);distributions=[declaration.get('distribution','')]; decode_manifest=folder/'decode_backends.json'
    if decode_manifest.exists():distributions+=json.loads(decode_manifest.read_text()).get(name,{}).get('cpu',{}).get('packages',[])
    if requirements.exists():
     from packaging.requirements import Requirement
     distributions += [Requirement(x).name for x in json.loads(requirements.read_text()).get(name,{}).get('packages',[])]
    for dist in filter(None,distributions):
     try:packages[dist]=md.version(dist)
     except md.PackageNotFoundError:packages[dist]='unavailable'
    spec=dict(dataset=filename,input_sha256=input_hash,bytes=len(data),codec=name,mode=mode,source=source,packages=packages,framework=common,environment=environment,block_bytes=block_bytes,cpu_workers=workers,decode_batch=group_size,timing=TIMING_VERSION,cpu_decoder=description)
    job=fingerprint(spec); (out/('job_'+job+'.json')).write_text(json.dumps(spec,sort_keys=True));total_blocks=(len(data)+block_bytes-1)//block_bytes
    # Resume identity is always the JSONL job fingerprint, never result.csv.
    completed={(path.stem,name,i):r for (key,i),r in done.items() if key==job}
    print('JOB',path.stem,name,mode,'resume',len(completed),'/',total_blocks,flush=True)
    decoder=BatchDecoder(route) if gpu_decode else None;warmed=set()
    encoded=encoded_blocks(data,block_bytes,route,completed,path.stem,1 if route.mode=='GPU-GPU' else workers)
    with journal.open('a',encoding='utf-8') as log:
     while True:
      batch=list(islice(encoded,group_size))
      if not batch:break
      if decoder:
       restored,seconds,kind=decoder.submit([x[2] for x in batch],[len(x[1]) for x in batch]).finish()
      else:
       decode=cpu_decoder or route.decode_gpu;index,raw,frame,encode_s=batch[0]
       if len(raw) not in warmed:
        assert decode(frame,len(raw))==raw;warmed.add(len(raw))
       start=time.perf_counter();restored=[decode(frame,len(raw))];seconds=time.perf_counter()-start;kind='cpu-single'
      raw_total=sum(len(x[1]) for x in batch);records=[]
      for item,actual in zip(batch,restored):
       index,raw,frame,encode_s=item
       if actual!=raw:raise RuntimeError('Lossless verification failed: '+name+' '+mode+' '+str(index))
       r=dict(block=index,raw_bytes=len(raw),compressed_bytes=len(frame),encode_seconds=encode_s,decode_seconds=seconds*len(raw)/raw_total,decode_batch_kind=kind)
       records.append(r);done[(job,index)]=r
      log.write(json.dumps(dict(job=job,records=records),separators=(',',':'))+'\n');log.flush();os.fsync(log.fileno())
    records=[done[(job,i)] for i in range(total_blocks)];raw=sum(x['raw_bytes'] for x in records);packed=sum(x['compressed_bytes'] for x in records)
    rows.append(dict(dataset=path.stem,codec=name,execution_path=mode,job_fingerprint=job,block_bytes=block_bytes,cpu_encode_threads=workers,decode_batch_blocks=group_size,decode_batch_kind=records[0]['decode_batch_kind'],blocks=total_blocks,verified_blocks=total_blocks,raw_bytes=raw,compressed_bytes=packed,compression_ratio=raw/packed,encode_mib_s=raw/1048576/sum(x['encode_seconds'] for x in records),decode_mib_s=raw/1048576/sum(x['decode_seconds'] for x in records),timing_version=TIMING_VERSION,encode_timing_scope='gpu_codec_call_cuda_events' if mode.startswith('GPU-') else 'cpu_codec_call_perf_counter',decode_timing_scope='gpu_codec_call_cuda_events' if gpu_decode else 'cpu_codec_call_perf_counter'))
    temporary=out/'result.csv.tmp'
    with temporary.open('w',newline='',encoding='utf-8') as f:
     writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    os.replace(temporary,out/'result.csv')
 (out/'capabilities.json').write_text(json.dumps(capabilities,ensure_ascii=False,indent=2))
 write_png(out/'heatmap.png',rows,tuple(routes))
 from html_report import write_report
 write_report(out,None if args.validate_only else root/'benchmark/output/latest.html')
 print('COMPLETE',len(rows),out,flush=True)
if __name__=='__main__':main()
