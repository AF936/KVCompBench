"""Render a self-contained report from separate HTML, CSS and JS sources."""
from pathlib import Path
from datetime import datetime
import base64,csv,html,json,math,os,tempfile

TEMPLATE_DIRECTORY=Path(__file__).resolve().parent/'templates'
def load_template():
    template=(TEMPLATE_DIRECTORY/'report.html').read_text(encoding='utf-8')
    return template.replace('__REPORT_CSS__',(TEMPLATE_DIRECTORY/'report.css').read_text(encoding='utf-8')).replace('__REPORT_JS__',(TEMPLATE_DIRECTORY/'report.js').read_text(encoding='utf-8'))
TEMPLATE=load_template()


def atomic_write(path, text):
    path=Path(path)
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,delete=False,suffix='.tmp') as f:
        temporary=Path(f.name)
        f.write(text)
    try: os.replace(temporary,path)
    finally:
        if temporary.exists(): temporary.unlink()

def codec_notes(directory):
    directory=Path(directory).resolve()
    notes={}
    if not directory.exists():return notes
    for manifest in directory.glob('*/codec.json'):
        sidecar=manifest.parent/'benchmark_notes.json'
        if not sidecar.exists():continue
        if not manifest.resolve().is_relative_to(directory) or not sidecar.resolve().is_relative_to(directory):
            raise ValueError('Codec notes path escapes compressors directory')
        declared={c['name'] for c in json.loads(manifest.read_text(encoding='utf-8'))['codecs']}
        content=json.loads(sidecar.read_text(encoding='utf-8'))
        if not isinstance(content,dict) or set(content)-declared:raise ValueError('Notes must reference declared codec names')
        for name,languages in content.items():
            if not isinstance(languages,dict) or not all(isinstance(v,str) for v in languages.values()):
                raise ValueError('Codec notes must contain language strings')
            notes[name]=languages.get('zh') or languages.get('en') or ''
    return notes

def write_report(output, latest=None, compressors=None):
    output=Path(output)
    csv_text=(output/'result.csv').read_text(encoding='utf-8')
    rows=list(csv.DictReader(csv_text.splitlines()))
    if not rows: raise ValueError('Empty benchmark results')
    notes=codec_notes(compressors or Path(__file__).resolve().parent/'compressors')
    for r in rows:
        r['codec_note']=notes.get(r['codec'],'')
        if r.get('timing_version','').startswith('codec_call_v3'):
            r['timing_note']='codec_call_v3：同一编码帧复用 CPU/GPU 解码；CPU 为 API 调用计时，GPU 为实际执行流 CUDA Event 计时；排除初始化、预校准、预热、传输、封装和校验。页面吞吐另计准备、封装和传输。'
        elif r.get('timing_version')=='codec_call_v2':
            r['timing_note']='codec_call_v2：CPU 调用边界计时；GPU 输入已驻留显存，使用 CUDA Event 测量实际执行流上的调用至计算完成。排除预热、H2D/D2H、结果封装与校验；调用内部必要处理和分配仍属于该 API 窗口。'
        else:
            r['timing_note']='历史端到端口径：包含当时框架的调用、输入准备及 GPU 传输等开销，不属于 codec_call_v2。'
        for extra in ('encode_page_mib_s','decode_page_mib_s'):
            if extra in r:r[extra]=float(r[extra])
        for key in ['compression_ratio','encode_mib_s','decode_mib_s','block_bytes','blocks','raw_bytes','compressed_bytes','verified_blocks']:
            r[key]=float(r[key])
            if not math.isfinite(r[key]): raise ValueError('Nonfinite result')
        if r['blocks']!=r['verified_blocks']: raise ValueError('Incomplete verification')
    summary='；'.join(html.escape(dataset)+': '+str(int(next(r['block_bytes'] for r in rows if r['dataset']==dataset)/1024))+' KiB / 块' for dataset in dict.fromkeys(r['dataset'] for r in rows))
    safe_json=lambda value:json.dumps(value,ensure_ascii=False).replace('<',chr(92)+'u003c')
    values={'__ROWS__':safe_json(rows),'__CSV__':safe_json(csv_text),'__PNG__':base64.b64encode((output/'heatmap.png').read_bytes()).decode(),'__COUNT__':str(len(rows)),'__BLOCKS__':format(int(sum(r['verified_blocks'] for r in rows)),','),'__SUMMARY__':summary,'__SOURCE__':html.escape(output.name+'/result.csv'),'__TIME__':datetime.now().astimezone().isoformat(timespec='seconds')}
    text=TEMPLATE.replace('__UNAVAILABLE__',safe_json(json.loads((output/'unavailable_codecs.json').read_text(encoding='utf-8')) if (output/'unavailable_codecs.json').exists() else {}))
    if all(str(r.get('validation_only','')).lower()=='true' for r in rows):text=text.replace('全量压缩测试','小样本兼容性验证')
    for key,value in values.items(): text=text.replace(key,value)
    atomic_write(output/'heatmap.html',text)
    if latest is not None: atomic_write(latest,text)
    return output/'heatmap.html'

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    parser.add_argument('--latest',type=Path)
    args=parser.parse_args()
    print(write_report(args.output,args.latest))
