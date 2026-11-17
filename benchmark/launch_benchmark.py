"""Launch the benchmark with the interpreter selected by local settings."""
from pathlib import Path
import configparser,os,re,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
def main():
    args=sys.argv[1:];settings=ROOT/'benchmark/settings.txt'
    if '--settings' in args:
        i=args.index('--settings')
        if i+1==len(args):raise ValueError('--settings requires a path')
        settings=Path(args[i+1])
        if not settings.is_absolute():settings=ROOT/settings
    if not settings.is_file():raise SystemExit('Create benchmark/settings.txt from benchmark/settings.example.txt first.')
    c=configparser.ConfigParser(interpolation=None);c.read(settings,encoding='utf-8');name=c.get('environment','name')
    if not re.fullmatch(r'[a-z0-9_-]+',name):raise ValueError('Invalid environment name')
    python=ROOT/'.venvs'/name/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    if not python.is_file():raise SystemExit('Environment missing: '+str(python))
    return subprocess.call([str(python),'-u',str(ROOT/'benchmark/run_full_cpu_encode_gpu_decode.py'),*args],cwd=ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
if __name__=='__main__':raise SystemExit(main())
