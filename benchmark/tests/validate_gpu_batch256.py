from pathlib import Path
import json,sys
root=Path(__file__).resolve().parents[2];b=root/'benchmark';sys.path[:0]=[str(b),str(b/'compression_framework')]
from benchmark_routes import discover_routes
from codec_timing import Encoder
from batch_decode import BatchDecoder
from kv_pages import Pages,dataset_configs
routes=discover_routes(b/'compressors');results=[]
for filename,config in dataset_configs(b/'settings.txt').items():
    pages=Pages(b/'input'/filename,config);source=[pages[0],pages[1],pages[pages.per_outer-1]]
    for name in ('splitzip','nvcomp-lz4'):
        runtime=routes[name].runtime();encoder=Encoder(runtime,source[0]);
        if hasattr(getattr(runtime,'codec',None),'calibrate'):
            def forbidden(*a,**k):raise AssertionError('Calibration must not run during encoding')
            runtime.codec.calibrate=forbidden
        frames=[encoder.measure(raw)[0] for raw in source]
        raw=[source[i%3] for i in range(256)];compressed=[frames[i%3] for i in range(256)]
        decoder=BatchDecoder(runtime);outputs,seconds,kind=decoder.submit(compressed,[len(x) for x in raw]).finish()
        assert outputs==raw
        results.append(dict(dataset=filename,codec=name,frames=256,raw_bytes=sum(map(len,raw)),decode_seconds=seconds,kind=kind,tail_bytes=len(source[-1]),verified=True))
        print(json.dumps(results[-1]),flush=True)
(b/'output/review_fix_final_validation/batch256_validation.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
