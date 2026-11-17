"""CPU-only repository contracts; no experimental input or GPU required."""
from pathlib import Path
import configparser,importlib.util,sys,unittest,json,tempfile,types
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
B=ROOT/'benchmark';sys.path.insert(0,str(B/'compression_framework'))
from benchmark_routes import discover_routes
class RepositoryLayout(unittest.TestCase):
    def test_core_cpu_roundtrips(self):
        routes=discover_routes(B/'compressors',codec_set='core',execution_device='cpu')
        self.assertTrue(routes)
        raw=(b'\x00\x80\x3f\x80'*1024)+b'\x01\x02'
        for name,route in routes.items():
            with self.subTest(codec=name):
                self.assertEqual(route.mode,'CPU-CPU');self.assertIsNone(route.gpu)
                self.assertNotIn('packages',route.options)
                runtime=route.runtime();runtime.setup(raw)
                frame=runtime.finish_encode(runtime.encode(runtime.prepare_encode(raw)),len(raw))
                self.assertEqual(runtime.cpu_decoder()(frame,len(raw)),raw)
    def test_profile_validation(self):
        with self.assertRaises(ValueError):discover_routes(B/'compressors',codec_set='unknown')
        with self.assertRaises(ValueError):discover_routes(B/'compressors',execution_device='unknown')
    def test_cli_accepts_repository_example(self):
        from cli import settings
        config,_=settings(B/'settings.example.txt')
        self.assertEqual(config.get('benchmark','codec_set'),'core')
        self.assertEqual(config.getint('benchmark','decode_batch_blocks'),256)
    def test_declared_routes_load_the_gpu_registry_lazily(self):
        module=types.ModuleType('gpu_codec');calls=[]
        module.discover_gpu_codecs=lambda directory:calls.append(directory) or {}
        with patch.dict(sys.modules,{'gpu_codec':module}):
            routes=discover_routes(B/'compressors',codec_set='core',execution_device='declared')
        self.assertTrue(routes);self.assertEqual(len(calls),1)
    def test_new_package_uses_its_own_tier(self):
        with tempfile.TemporaryDirectory(dir=B/'output') as directory:
            root=Path(directory);p=root/'new_package';p.mkdir()
            (p/'codec.json').write_text(json.dumps({'schema_version':1,'codecs':[{'name':'fixture-codec','version':'1.0','kind':'python','file':'backend.py','compress':'encode','decompress':'decode','interface':'context'}]}))
            (p/'backend.py').write_text('def encode(raw, metadata):return raw\ndef decode(frame, size, metadata):return frame\n')
            self.assertIn('fixture-codec',discover_routes(root,execution_device='cpu'))
            self.assertNotIn('fixture-codec',discover_routes(root,codec_set='core',execution_device='cpu'))
            (p/'package.json').write_text(json.dumps({'tier':'core'}))
            self.assertIn('fixture-codec',discover_routes(root,codec_set='core',execution_device='cpu'))
    def test_example_and_templates(self):
        c=configparser.ConfigParser();c.read(B/'settings.example.txt')
        self.assertEqual(c.get('benchmark','codec_set'),'core')
        self.assertEqual(c.get('benchmark','execution_device'),'cpu')
        ignores=(B.parent/'.gitignore').read_text(encoding='utf-8')
        self.assertIn('/.repository_archive/',ignores)
        self.assertIn('/benchmark/settings.txt',ignores)
        self.assertIn('/.venvs/',ignores)
        for name in ('report.html','report.css','report.js'):self.assertTrue((B/'templates'/name).is_file())
        report=(B/'html_report.py').read_text(encoding='utf-8')
        self.assertNotIn('<script>',report);self.assertNotIn('def patch(',report)
if __name__=='__main__':unittest.main()
