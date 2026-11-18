
const rows=__ROWS__;
const csv=__CSV__; const omitted=__UNAVAILABLE__;document.querySelector('#unavailable').textContent=Object.entries(omitted).map(([name,reason])=>name+': '+reason).join('\n')||'无 / None';
const metric=document.querySelector('#metric'),route=document.querySelector('#route'),table=document.querySelector('#heatmap');
const decoder=document.querySelector('#decoder');
const storage=document.querySelector('#storage');
for(const dtype of new Set(rows.map(r=>r.storage_dtype||'bf16'))){const option=document.createElement('option');option.value=dtype;option.textContent=({bf16:'BF16',fp8_e4m3fn:'FP8 E4M3FN',fp8_e5m2:'FP8 E5M2'})[dtype]||dtype;storage.append(option)}
function render(){
const selected=rows.filter(r=>(route.value==='all'||r.execution_path===route.value)&&(storage.value==='all'||(r.storage_dtype||'bf16')===storage.value)),data=selected.filter(r=>r.execution_path.split('-').pop()===decoder.value),names=[...new Set(selected.map(r=>r.codec))],datasets=[...new Set(selected.map(r=>r.dataset))];
const max=Math.max(0,...data.map(r=>r[metric.value]).filter(Number.isFinite));
table.replaceChildren();const head=table.createTHead().insertRow();let h=document.createElement('th');h.textContent='数据集 / Codec';head.append(h);
for(const name of names){const th=document.createElement('th');th.textContent=name;const small=document.createElement('small');const info=data.find(r=>r.codec===name)||rows.find(r=>r.codec===name);small.textContent=(data.some(r=>r.codec===name)?info.execution_path:'/')+(info.decode_batch_blocks?' · '+info.decode_batch_blocks+' 块/批':'');th.append(small);th.title=rows.find(r=>r.codec===name).codec_note||'';head.append(th)}
const body=table.createTBody();for(const dataset of datasets){const tr=body.insertRow(),th=document.createElement('th');th.textContent=dataset.replace('_bf16','');const small=document.createElement('small');small.textContent=(rows.find(r=>r.dataset===dataset).block_bytes/1024)+' KiB / 块';th.append(small);tr.append(th);
for(const name of names){const r=data.find(r=>r.dataset===dataset&&r.codec===name),td=tr.insertCell();if(r&&!Number.isFinite(r[metric.value])){td.textContent='/';continue;}if(!r){td.textContent='/';td.title='该解码端没有对应方案或实测结果';continue;}const v=r[metric.value],t=max>0?v/max:0;td.textContent=v.toFixed(2)+(metric.value==='compression_ratio'?'×':'');td.style.background=`hsl(${200-40*t} ${45+25*t}% ${18+20*t}%)`;td.tabIndex=0;
let text=`${dataset} · ${name} · ${r.execution_path}
压缩比：${r.compression_ratio.toFixed(4)}× ｜ 编码：${r.encode_mib_s.toFixed(2)} MiB/s ｜ 解码：${r.decode_mib_s.toFixed(2)} MiB/s
原始：${r.raw_bytes.toLocaleString()} 字节 ｜ 压缩后：${r.compressed_bytes.toLocaleString()} 字节 ｜ 无损校验：${r.verified_blocks.toLocaleString()} / ${r.blocks.toLocaleString()} 块`;
if(r.storage_dtype)text+='\n存储格式：'+r.storage_dtype+' ｜ 页面格式：'+r.page_format+' ｜ data/scale 字节：'+r.data_bytes+' / '+r.scale_bytes+' ｜ 量化来源：'+r.quantization_origin;
if(r.quantization_metrics&&r.quantization_metrics!=='{}')text+='\n量化误差元数据：'+r.quantization_metrics;
if(r.codec_note)text+='\nCodec 备注：'+r.codec_note;text+='\n计时口径：'+r.timing_note;td.title=text;td.setAttribute('aria-label',text);for(const e of ['mouseenter','focus','click'])td.addEventListener(e,()=>document.querySelector('#detail').textContent=text);
}}
document.querySelector('#scale').textContent=`当前色标范围：0 — ${max.toFixed(2)} ${metric.value==='compression_ratio'?'×':'MiB/s'}。`;
}metric.onchange=route.onchange=decoder.onchange=storage.onchange=render;render();
document.querySelector('#download').onclick=()=>{const url=URL.createObjectURL(new Blob(['﻿'+csv],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download='result.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
