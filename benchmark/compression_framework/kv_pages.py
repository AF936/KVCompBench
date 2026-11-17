"""Explicit tensor-axis KV pages and a separate generic flat-block baseline."""
import configparser,itertools,json,math
from pathlib import Path
class Pages:
    def __init__(self,path,config):
        import torch
        self.torch=torch;value=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        self.metadata=value if isinstance(value,dict) else {};self.tensor=value['data'] if isinstance(value,dict) else value
        if not isinstance(self.tensor,torch.Tensor) or self.tensor.dtype!=torch.bfloat16:raise ValueError('Expected BF16 tensor')
        self.config=dict(config);self.layout=config['layout'];self.shape=list(self.tensor.shape)
        self.dataset_metadata={k:self.metadata[k] for k in ('format','conversion','key_value_shared','prequant_bf16_exactly_recoverable','source_fp8_dtype') if k in self.metadata}
        if self.layout=='flat':
            self.block_bytes=int(config['block_bytes'])
            if self.block_bytes<2 or self.block_bytes%2:raise ValueError('BF16 block_bytes must be a positive multiple of 2')
            self.flat=self.tensor.contiguous().reshape(-1).view(torch.uint8).numpy();self.count=(len(self.flat)+self.block_bytes-1)//self.block_bytes
        else:
            axes=config['axes'].split(',');axes=[x.strip() for x in axes]
            if len(axes)!=len(self.shape) or len(set(axes))!=len(axes) or not {'layer','token'}<=set(axes):raise ValueError('Explicit unique layer/token axes required')
            if any(x not in {'layer','kv','batch','head','token','dim','latent'} for x in axes):raise ValueError('Unsupported axis name')
            if config.get('kv_semantics')=='shared' and 'kv' in axes:raise ValueError('Shared latent cannot have a fabricated K/V axis')
            self.axes=axes;self.token_axis=axes.index('token');self.tokens=int(config['page_tokens'])
            if self.tokens<=0:raise ValueError('page_tokens must be positive')
            self.outer_axes=[i for i,x in enumerate(axes) if x in ('layer','batch')]
            self.outer=list(itertools.product(*(range(self.shape[i]) for i in self.outer_axes)))
            self.per_outer=(self.shape[self.token_axis]+self.tokens-1)//self.tokens
            self.count=len(self.outer)*self.per_outer
            self.block_bytes=2*self.tokens*math.prod(n for i,n in enumerate(self.shape) if i not in self.outer_axes+[self.token_axis])
    def __len__(self):return self.count
    def descriptor(self,index):
        if not 0<=index<len(self):raise IndexError(index)
        if self.layout=='flat':return dict(index=index,byte_start=index*self.block_bytes,layout='flat')
        outer,page=divmod(index,self.per_outer);start=page*self.tokens;stop=min(start+self.tokens,self.shape[self.token_axis])
        return dict(index=index,layout='kv',coordinates={self.axes[i]:v for i,v in zip(self.outer_axes,self.outer[outer])},token_start=start,token_stop=stop,axes=self.axes,kv_semantics=self.config.get('kv_semantics','separate'))
    def __getitem__(self,index):
        d=self.descriptor(index)
        if self.layout=='flat':return self.flat[index*self.block_bytes:(index+1)*self.block_bytes].tobytes()
        slices=[slice(None)]*len(self.shape)
        for i,v in zip(self.outer_axes,self.outer[index//self.per_outer]):slices[i]=slice(v,v+1)
        slices[self.token_axis]=slice(d['token_start'],d['token_stop'])
        return self.tensor[tuple(slices)].contiguous().view(self.torch.uint8).numpy().tobytes()

def dataset_configs(settings):
    c=configparser.ConfigParser(interpolation=None);c.read(settings,encoding='utf-8')
    legacy=dict(c['benchmark_datasets'])
    result={}
    for filename,block in legacy.items():
        if Path(filename).name!=filename:raise ValueError('Dataset must be a filename')
        section='dataset:'+filename
        values=dict(c[section]) if c.has_section(section) else dict(layout='flat')
        if values.get('layout') not in ('kv','flat'):raise ValueError('layout must be kv or flat')
        values['block_bytes']=int(block);result[filename]=values
    return result

class PageSubset:
    def __init__(self,base,indices):self.base=base;self.indices=indices
    def __len__(self):return len(self.indices)
    def __getattr__(self,key):return getattr(self.base,key)
    def __getitem__(self,index):return self.base[self.indices[index]]
    def descriptor(self,index):return dict(self.base.descriptor(self.indices[index]),validation_source_index=self.indices[index])

def validation_subset(pages):
    return PageSubset(pages,sorted(set([0,min(1,len(pages)-1),pages.per_outer-1 if pages.layout=='kv' else len(pages)-1])))
