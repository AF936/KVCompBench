import ast,csv,hashlib,importlib.util,json,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('runner',ROOT/'benchmark/run_full_cpu_encode_gpu_decode.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
class ReleaseTests(unittest.TestCase):
 def test_codec_local_identities(self):
  a=dict(codec='A',source={'api.py':'v1'},input_sha256='data',mode='CPU-GPU')
  key=runner.fingerprint(a)
  runner.fingerprint(dict(codec='B',source={'api.py':'v1'}))
  self.assertEqual(key,runner.fingerprint(a))
  self.assertNotEqual(key,runner.fingerprint(dict(a,source={'api.py':'v2'})))
  self.assertNotEqual(key,runner.fingerprint(dict(a,mode='CPU-CPU')))
  self.assertNotEqual(key,runner.fingerprint(dict(a,input_sha256='changed')))
 def test_journal_keeps_paths_separate_and_recovers_tail(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'progress.jsonl'
   records=[dict(job='cpu',records=[{'block':0}]),dict(job='gpu',records=[{'block':0}])]
   valid=''.join(json.dumps(x)+'\n' for x in records).encode()
   p.write_bytes(valid+b'{"job":')
   self.assertEqual(set(runner.load_journal(p)),{('cpu',0),('gpu',0)})
   self.assertEqual(p.read_bytes(),valid)
   self.assertEqual(len(list(Path(d).glob('*.jsonpart'))),1)
   with p.open('ab') as f:f.write((json.dumps(dict(job='cpu',records=[{'block':1}]))+'\n').encode())
   self.assertEqual(len(runner.load_journal(p)),3)
 def test_complete_final_record_without_newline(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'progress.jsonl';p.write_text(json.dumps(dict(job='a',records=[{'block':0}])))
   self.assertEqual(len(runner.load_journal(p)),1);self.assertTrue(p.read_bytes().endswith(b'\n'))
 def test_source_integrity(self):
  for manifest in (ROOT/'benchmark/compressors').glob('*/source_hashes.json'):
   for rel,expected in json.loads(manifest.read_text()).items():
    p=manifest.parent/rel;self.assertTrue(p.is_file(),str(p));self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),expected,str(p))
 def test_python_syntax(self):
  for p in (ROOT/'benchmark').rglob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
 def test_reference_results(self):
  with (ROOT/'examples/results/rtx5090_bf16_20261115/result.csv').open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
  self.assertEqual(len(rows),80)
  self.assertTrue(all(r['blocks']==r['verified_blocks'] and r['timing_version']=='codec_call_v2' for r in rows))
  self.assertEqual(len({(r['dataset'],r['codec'],r['execution_path']) for r in rows}),80)
if __name__=='__main__':unittest.main()
