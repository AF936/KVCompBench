
import json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'compression_framework')]
import torch
from kv_pages import Pages
from codec_timing import Encoder
from run_full_cpu_encode_gpu_decode import load_journal
class Contracts(unittest.TestCase):
    def test_kv_boundaries_and_lossless_slicing(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'data.pt';t=torch.arange(2*2*1*3*7*4).reshape(2,2,1,3,7,4).to(torch.bfloat16);torch.save(t,p)
            pages=Pages(p,dict(layout='kv',axes='layer,kv,batch,head,token,dim',page_tokens=3,kv_semantics='separate',block_bytes=2))
            self.assertEqual(len(pages),6);self.assertEqual(pages.block_bytes,144)
            rebuilt=torch.empty_like(t)
            for i in range(len(pages)):
                d=pages.descriptor(i);l=d['coordinates']['layer'];a=d['token_start'];z=d['token_stop']
                expected=t[l:l+1,:,:,:,a:z,:].contiguous()
                self.assertEqual(pages[i],expected.view(torch.uint8).numpy().tobytes())
                rebuilt[l:l+1,:,:,:,a:z,:]=torch.frombuffer(bytearray(pages[i]),dtype=torch.bfloat16).reshape(expected.shape)
            self.assertTrue(torch.equal(t.view(torch.int16),rebuilt.view(torch.int16)))
            self.assertEqual(sum(len(pages[i]) for i in range(len(pages))),t.numel()*2)
    def test_shared_latent_and_flat(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'data.pt';torch.save({'data':torch.ones((2,5,4),dtype=torch.bfloat16)},p)
            pages=Pages(p,dict(layout='kv',axes='layer,token,latent',page_tokens=3,kv_semantics='shared',block_bytes=16))
            self.assertEqual([len(pages[i]) for i in range(len(pages))],[24,16,24,16])
            flat=Pages(p,dict(layout='flat',block_bytes=16));self.assertEqual(len(flat),5)
            with self.assertRaises(ValueError):Pages(p,dict(layout='kv',axes='layer,kv,token',page_tokens=3,kv_semantics='shared',block_bytes=16))
    def test_setup_and_warmup_excluded(self):
        class Runtime:
            stream=None;route=type('Route',(),{'mode':'CPU-CPU'})()
            def setup(self,sample):self.setups=1;self.calls=0
            def prepare_encode(self,raw):return raw
            def encode(self,raw):self.calls+=1;return raw
            def finish_encode(self,value,size):return value
        runtime=Runtime();encoder=Encoder(runtime,b'abc');encoder.measure(b'abc');encoder.measure(b'abc')
        self.assertEqual(runtime.setups,1);self.assertEqual(runtime.calls,3)
    def test_journal_tail_repair_and_csv_independence(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'progress.jsonl';line=json.dumps(dict(type='decode',job='j',records=[dict(block=0)]))+'\n'
            p.write_text(line+'{"incomplete":',encoding='utf-8');done,enc=load_journal(p)
            self.assertIn(('j',0),done);self.assertEqual(p.read_text(),line)
            with p.open('a') as f:f.write(json.dumps(dict(type='encode',job='e',records=[dict(block=0)]))+'\n')
            (Path(tmp)/'result.csv').write_text('invalid CSV does not control resume')
            done,enc=load_journal(p);self.assertIn(('e',0),enc)
    def test_external_plugin_without_runner_branch(self):
        from benchmark_routes import discover_routes
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'new_codec';folder.mkdir()
            (folder/'api.py').write_text("def compress(x,m):return x[::-1]\ndef decompress(x,n,m):return x[::-1]\n")
            (folder/'codec.json').write_text(json.dumps(dict(schema_version=1,codecs=[dict(name='new-reverse-example',kind='python',version='1',file='api.py',interface='context',compress='compress',decompress='decompress',dtypes=['bf16'])])))
            route=discover_routes(Path(tmp))['new-reverse-example'];runtime=route.runtime();encoder=Encoder(runtime,b'1234');frame,*_=encoder.measure(b'1234')
            self.assertEqual(runtime.cpu_decoder()(frame,4),b'1234')
    def test_runner_has_no_codec_name_dispatch(self):
        for name in ('benchmark_routes.py','codec_timing.py','batch_decode.py'):
            text=(ROOT/'compression_framework'/name).read_text()
            for codec in ('splitzip','nvcomp','openzl','zipnn'):self.assertNotIn(codec,text.lower())
if __name__=='__main__':unittest.main(verbosity=2)
