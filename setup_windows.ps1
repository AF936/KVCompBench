$ErrorActionPreference = 'Stop'
$setup = Join-Path $PSScriptRoot 'benchmark/compression_framework/setup_environment.py'
if (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3 $setup @args
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    & python $setup @args
} else {
    throw 'Install 64-bit Python 3.10+ with pip and venv, then rerun this script.'
}
exit $LASTEXITCODE
