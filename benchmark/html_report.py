"""Offline HTML report generation; no GPU dependency."""
from pathlib import Path
from datetime import datetime
import base64,csv,html,json,math,os,tempfile

TEMPLATE='<!doctype html>\n<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">\n<title>BF16 KV Cache · 全量压缩测试</title>\n<style>\n:root{color-scheme:dark;font-family:system-ui,"Microsoft YaHei",sans-serif;background:#0d141f;color:#deeaf7}body{margin:0;padding:32px}main{max-width:1600px;margin:auto}h1{font-size:28px;margin:0 0 12px}p{color:#aebed0;line-height:1.8}.cards{display:flex;gap:14px;flex-wrap:wrap;margin:24px 0}.card{background:#172332;border:1px solid #2a3a4c;border-radius:12px;padding:16px 24px}.card strong{display:block;font-size:24px;color:#56e2c1}label{display:inline-flex;align-items:center;gap:10px;margin:0 20px 16px 0}select,button{padding:10px;border-radius:8px;background:#192a3c;color:#e0edf9;border:1px solid #425970;font:inherit}.scroll{overflow:auto;border:1px solid #304357;border-radius:12px;max-height:70vh}table{border-collapse:separate;border-spacing:0;width:100%;font-size:14px}th,td{padding:13px 17px;border-bottom:1px solid #304357;text-align:center;white-space:nowrap}thead th{position:sticky;top:0;background:#152333;z-index:2}th:first-child{position:sticky;left:0;background:#152333;z-index:3;text-align:left}thead th:first-child{z-index:4}tbody th{min-width:150px}th small{display:block;margin-top:8px;color:#9eafc2;font-weight:400}td{font-variant-numeric:tabular-nums;font-weight:600;color:white}td:focus,td:hover{outline:2px solid #fff;outline-offset:-3px}.legend{display:flex;gap:14px;align-items:center;margin:18px 0}.gradient{width:200px;height:12px;background:linear-gradient(90deg,hsl(200 45% 18%),hsl(160 70% 38%));border-radius:9px}#detail{padding:18px;background:#152333;border:1px solid #304357;border-radius:10px;min-height:52px;line-height:1.8;white-space:pre-line}details{margin-top:28px}summary{cursor:pointer;color:#aee7db}img{width:100%;min-width:1500px}footer{margin-top:24px;color:#98aabe;font-size:13px}@media(max-width:600px){body{padding:16px}h1{font-size:23px}}\n</style><main>\n<h1>BF16 KV Cache · 全量压缩测试</h1>\n<p>__SUMMARY__</p><div class="cards"><div class="card"><strong>__COUNT__</strong>数据集 × 配置结果</div><div class="card"><strong>__BLOCKS__</strong>已验证数据块</div></div>\n<label>显示指标<select id="metric"><option value="compression_ratio">压缩比（原始 / 压缩）</option><option value="encode_mib_s">编码吞吐量（MiB/s）</option><option value="decode_mib_s">解码吞吐量（MiB/s）</option></select></label>\n<label>执行路径<select id="route"><option value="all">全部</option><option>CPU-CPU</option><option>CPU-GPU</option><option>GPU-GPU</option><option>GPU-CPU</option></select></label>\n<button id="download">下载结果 CSV</button>\n<div class="scroll"><table id="heatmap" aria-label="全量测试热力图"></table></div>\n<div class="legend"><span>较低</span><span class="gradient"></span><span>较高（当前指标共用线性色标）</span></div>\n<p id="scale"></p><div id="detail" aria-live="polite">悬停、点击或使用 Tab 聚焦单元格，查看完整指标。</div>\n<p>CPU-GPU 表示 CPU 编码、GPU 解码；GPU-GPU 表示 GPU 编解码；CPU-CPU 表示 CPU 编解码。吞吐量包含当前框架的调用、初始化与传输等开销，并非仅测 GPU 内核。SplitZip 包含逐块校准和封装；Huff0 超过 128 KiB 时内部自动分段，分段开销计入压缩大小。压缩比小于 1 表示输出膨胀。不同模式的编码结果使用各自格式。</p>\n<details><summary>查看原始 PNG 热力图</summary><div class="scroll"><img alt="原始全量热力图" src="data:image/png;base64,__PNG__"></div></details>\n<footer>来源：__SOURCE__ · 报告生成：__TIME__ · 数据与图片内嵌，可离线打开。</footer>\n</main><script>\nconst rows=__ROWS__;\nconst csv=__CSV__;\nconst metric=document.querySelector(\'#metric\'),route=document.querySelector(\'#route\'),table=document.querySelector(\'#heatmap\');\nfunction render(){\nconst data=rows.filter(r=>route.value===\'all\'||r.execution_path===route.value),names=[...new Set(data.map(r=>r.codec))],datasets=[...new Set(data.map(r=>r.dataset))];\nconst max=Math.max(...data.map(r=>r[metric.value]));\ntable.replaceChildren();const head=table.createTHead().insertRow();let h=document.createElement(\'th\');h.textContent=\'数据集 / Codec\';head.append(h);\nfor(const name of names){const th=document.createElement(\'th\');th.textContent=name;const small=document.createElement(\'small\');small.textContent=data.find(r=>r.codec===name).execution_path;th.append(small);head.append(th)}\nconst body=table.createTBody();for(const dataset of datasets){const tr=body.insertRow(),th=document.createElement(\'th\');th.textContent=dataset.replace(\'_bf16\',\'\');const small=document.createElement(\'small\');small.textContent=(data.find(r=>r.dataset===dataset).block_bytes/1024)+\' KiB / 块\';th.append(small);tr.append(th);\nfor(const name of names){const r=data.find(r=>r.dataset===dataset&&r.codec===name),td=tr.insertCell(),v=r[metric.value],t=max>0?v/max:0;td.textContent=v.toFixed(2)+(metric.value===\'compression_ratio\'?\'×\':\'\');td.style.background=`hsl(${200-40*t} ${45+25*t}% ${18+20*t}%)`;td.tabIndex=0;\nconst text=`${dataset} · ${name} · ${r.execution_path}\n压缩比：${r.compression_ratio.toFixed(4)}× ｜ 编码：${r.encode_mib_s.toFixed(2)} MiB/s ｜ 解码：${r.decode_mib_s.toFixed(2)} MiB/s\n原始：${r.raw_bytes.toLocaleString()} 字节 ｜ 压缩后：${r.compressed_bytes.toLocaleString()} 字节 ｜ 无损校验：${r.verified_blocks.toLocaleString()} / ${r.blocks.toLocaleString()} 块`;\ntd.title=text;td.setAttribute(\'aria-label\',text);for(const e of [\'mouseenter\',\'focus\',\'click\'])td.addEventListener(e,()=>document.querySelector(\'#detail\').textContent=text);\n}}\ndocument.querySelector(\'#scale\').textContent=`当前色标范围：0 — ${max.toFixed(2)} ${metric.value===\'compression_ratio\'?\'×\':\'MiB/s\'}。`;\n}metric.onchange=route.onchange=render;render();\ndocument.querySelector(\'#download\').onclick=()=>{const url=URL.createObjectURL(new Blob([\'\ufeff\'+csv],{type:\'text/csv;charset=utf-8\'}));const a=document.createElement(\'a\');a.href=url;a.download=\'result.csv\';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};\n</script></html>'

def atomic_write(path, text):
    path=Path(path)
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,delete=False,suffix='.tmp') as f:
        temporary=Path(f.name)
        f.write(text)
    try: os.replace(temporary,path)
    finally:
        if temporary.exists(): temporary.unlink()

TEMPLATE=TEMPLATE.replace('small.textContent=data.find(r=>r.codec===name).execution_path;', "const info=data.find(r=>r.codec===name);small.textContent=info.execution_path+(info.decode_batch_blocks?' · '+info.decode_batch_blocks+' 块/批':'');")

_begin=TEMPLATE.index('<p>CPU-GPU')
_end=TEMPLATE.index('</p>',_begin)+4
TEMPLATE=TEMPLATE[:_begin]+'<p>CPU-GPU 表示 CPU 编码、GPU 解码；GPU-GPU 表示 GPU 编解码；CPU-CPU 表示 CPU 编解码。吞吐量统一按原始未压缩字节数除以对应计时时长计算，单位为 MiB/s。计时范围由结果记录的测试口径决定，不能将包含传输的端到端吞吐量当作纯 GPU 计算吞吐量。压缩比小于 1 表示输出膨胀。悬停列标题或选中单元格可查看该 codec 的独立备注。</p>'+TEMPLATE[_end:]
TEMPLATE=TEMPLATE.replace('const text=`${dataset}', 'let text=`${dataset}')
TEMPLATE=TEMPLATE.replace('td.title=text;', "if(r.codec_note)text+='"+chr(92)+"nCodec 备注：'+r.codec_note;td.title=text;")
TEMPLATE=TEMPLATE.replace('head.append(th)}', "th.title=data.find(r=>r.codec===name).codec_note||'';head.append(th)}")
TEMPLATE=TEMPLATE.replace('td.title=text;', "text+='"+chr(92)+"n计时口径：'+r.timing_note;td.title=text;")
TEMPLATE=TEMPLATE.replace('<footer>', '<details><summary>未运行的 codec / Unavailable codecs</summary><pre id="unavailable"></pre></details><footer>')
TEMPLATE=TEMPLATE.replace('const csv=__CSV__;', "const csv=__CSV__; const omitted=JSON.parse(rows[0].unavailable_codecs||'{}');document.querySelector('#unavailable').textContent=Object.entries(omitted).map(([name,reason])=>name+': '+reason).join('"+chr(92)+"n')||'无 / None';")

def patch(t):
 t=t.replace('<label>执行路径', '<label>解码端<select id="decoder"><option value="GPU">GPU 解码</option><option value="CPU">CPU 解码</option></select></label>\n<label>执行路径',1)
 t=t.replace("function render(){","const decoder=document.querySelector('#decoder');\nfunction render(){",1)
 old="const data=rows.filter(r=>route.value==='all'||r.execution_path===route.value),names=[...new Set(data.map(r=>r.codec))],datasets=[...new Set(data.map(r=>r.dataset))];"
 new="const selected=rows.filter(r=>route.value==='all'||r.execution_path===route.value),data=selected.filter(r=>r.execution_path.split('-').pop()===decoder.value),names=[...new Set(selected.map(r=>r.codec))],datasets=[...new Set(rows.map(r=>r.dataset))];"
 assert old in t;t=t.replace(old,new,1)
 t=t.replace('Math.max(...data.map','Math.max(0,...data.map')
 t=t.replace('const info=data.find(r=>r.codec===name);',"const info=data.find(r=>r.codec===name)||rows.find(r=>r.codec===name);")
 t=t.replace("small.textContent=info.execution_path+", "small.textContent=(data.some(r=>r.codec===name)?info.execution_path:'/')+")
 t=t.replace("th.title=data.find(r=>r.codec===name).codec_note", "th.title=rows.find(r=>r.codec===name).codec_note")
 t=t.replace('data.find(r=>r.dataset===dataset).block_bytes','rows.find(r=>r.dataset===dataset).block_bytes')
 old="td=tr.insertCell(),v=r[metric.value]"
 new="td=tr.insertCell();if(!r){td.textContent='/';td.title='该解码端没有对应方案或实测结果';continue;}const v=r[metric.value]"
 assert old in t;t=t.replace(old,new,1)
 t=t.replace('metric.onchange=route.onchange=render','metric.onchange=route.onchange=decoder.onchange=render')
 t=t.replace('<div class="scroll"><table', '<p>选择 CPU 或 GPU 解码端查看对应实测数据；没有对应方案或尚无实测结果时显示“/”。</p><div class="scroll"><table',1)
 return t

TEMPLATE=patch(TEMPLATE)

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
        if r.get('timing_version')=='codec_call_v2':
            r['timing_note']='codec_call_v2：CPU 调用边界计时；GPU 输入已驻留显存，使用 CUDA Event 测量实际执行流上的调用至计算完成。排除预热、H2D/D2H、结果封装与校验；调用内部必要处理和分配仍属于该 API 窗口。'
        else:
            r['timing_note']='历史端到端口径：包含当时框架的调用、输入准备及 GPU 传输等开销，不属于 codec_call_v2。'
        for key in ['compression_ratio','encode_mib_s','decode_mib_s','block_bytes','blocks','raw_bytes','compressed_bytes','verified_blocks']:
            r[key]=float(r[key])
            if not math.isfinite(r[key]): raise ValueError('Nonfinite result')
        if r['blocks']!=r['verified_blocks']: raise ValueError('Incomplete verification')
    summary='；'.join(html.escape(dataset)+': '+str(int(next(r['block_bytes'] for r in rows if r['dataset']==dataset)/1024))+' KiB / 块' for dataset in dict.fromkeys(r['dataset'] for r in rows))
    safe_json=lambda value:json.dumps(value,ensure_ascii=False).replace('<',chr(92)+'u003c')
    values={'__ROWS__':safe_json(rows),'__CSV__':safe_json(csv_text),'__PNG__':base64.b64encode((output/'heatmap.png').read_bytes()).decode(),'__COUNT__':str(len(rows)),'__BLOCKS__':format(int(sum(r['verified_blocks'] for r in rows)),','),'__SUMMARY__':summary,'__SOURCE__':html.escape(output.name+'/result.csv'),'__TIME__':datetime.now().astimezone().isoformat(timespec='seconds')}
    text=TEMPLATE
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
