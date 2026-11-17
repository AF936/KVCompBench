"""Unified cross-platform entry point for KVCompBench."""
from pathlib import Path
import argparse,configparser,os,platform,subprocess,sys,venv
ROOT=Path(__file__).resolve().parent
B=ROOT/'benchmark'
DEFAULT=B/'settings.txt'

def call(args,**kwargs):
    return subprocess.run([str(x) for x in args],cwd=ROOT,check=True,**kwargs)

def config_path(value):
    p=Path(value) if value else DEFAULT
    return p if p.is_absolute() else ROOT/p

def interpreter(settings):
    c=configparser.ConfigParser(interpolation=None);c.read(settings)
    explicit=c.get('environment','python',fallback='')
    if explicit:
        p=Path(explicit)
    else:
        name=c.get('environment','name')
        if Path(name).name!=name or name in ('.','..'):raise ValueError('Invalid environment name')
        p=ROOT/'.venvs'/name/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    if not p.is_file():raise SystemExit('Environment missing. Run: python kvbench.py setup')
    return p

def basic_settings():
    if not DEFAULT.exists():
        DEFAULT.write_text((B/'settings.example.txt').read_text(),encoding='utf-8')
    return DEFAULT

def setup_basic():
    name=platform.system().lower()+'-'+platform.machine().lower()+'-basic-cpu'
    env=ROOT/'.venvs'/name
    scratch=env/'tmp';scratch.mkdir(parents=True,exist_ok=True)
    variables=dict(os.environ,TMPDIR=str(scratch),TEMP=str(scratch),TMP=str(scratch),
                   PIP_CACHE_DIR=str(env/'pip-cache'),PYTHONDONTWRITEBYTECODE='1')
    py=env/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    if not py.exists():venv.EnvBuilder(with_pip=True,symlinks=False).create(env)
    call([py,'-m','pip','install','torch==2.9.1','--index-url','https://download.pytorch.org/whl/cpu'],env=variables)
    call([py,'-m','pip','install','-r',B/'requirements/core.txt'],env=variables)
    call([py,'-m','pip','check'],env=variables)
    c=configparser.ConfigParser(interpolation=None);c.read(basic_settings())
    c['environment']={'name':name}
    with DEFAULT.open('w',encoding='utf-8') as f:c.write(f)
    print('Basic CPU environment ready:',py)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    setup=sub.add_parser('setup',help='Install the basic CPU profile')
    setup.add_argument('--optional',action='store_true',help='Install the optional full CUDA/native profile (Linux x86-64)')
    run=sub.add_parser('run',help='Run core CPU codecs by default')
    run.add_argument('--settings')
    cli=sub.add_parser('cli',help='Generic compression CLI')
    cli.add_argument('--settings')
    sub.add_parser('codecs',help='List core and optional packages without loading dependencies')
    args,extra=parser.parse_known_args()
    if args.command=='setup':
        if args.optional:
            call([sys.executable,B/'compression_framework/setup_paper_codecs.py',*extra])
        else:
            if extra:parser.error('Unknown setup arguments: '+' '.join(extra))
            setup_basic()
    elif args.command=='codecs':
        if extra:parser.error('Unexpected arguments')
        import json
        for p in sorted((B/'compressors').glob('*/codec.json')):
            tier=json.loads((p.parent/'package.json').read_text()).get('tier','optional') if (p.parent/'package.json').exists() else 'optional'
            print(tier,p.parent.name,', '.join(x['name'] for x in json.loads(p.read_text())['codecs']))
    else:
        settings=config_path(args.settings) if args.settings else basic_settings()
        py=interpreter(settings)
        target=B/'run_full_cpu_encode_gpu_decode.py' if args.command=='run' else B/'compression_framework/cli.py'
        call([py,'-u','-B',target,'--settings',settings,*extra],env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
if __name__=='__main__':main()
