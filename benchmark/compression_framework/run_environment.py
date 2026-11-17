"""Run the framework using the environment selected in settings.txt."""
import configparser
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    args = sys.argv[1:]
    settings_path = ROOT/'benchmark/settings.txt'
    if '--settings' in args:
        index = args.index('--settings')
        if index + 1 >= len(args):raise ValueError('--settings requires a path')
        settings_path = Path(args[index + 1]).resolve()
    config = configparser.ConfigParser(interpolation=None)
    with settings_path.open(encoding='utf-8') as handle:config.read_file(handle)
    name = config.get('environment','name',fallback='').strip()
    if not re.fullmatch(r'[a-z0-9_-]+',name):
        raise ValueError('Run a setup script to select an environment in settings.txt')
    env = ROOT/'.venvs'/name
    if not env.resolve().is_relative_to((ROOT/'.venvs').resolve()):
        raise ValueError('Environment path escapes the project environment directory')
    python = env/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    if not python.is_file():raise FileNotFoundError(f'Environment is not installed on this machine: {env}. Run its setup script locally.')
    command = [str(python),str(Path(__file__).with_name('cli.py')),*args]
    return subprocess.call(command)


if __name__ == '__main__':
    raise SystemExit(main())
