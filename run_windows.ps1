$ErrorActionPreference = 'Stop'
$launcher = Join-Path $PSScriptRoot 'benchmark/compression_framework/run_environment.py'
if (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3 $launcher @args
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    & python $launcher @args
} else {
    throw 'Python 3.10+ is required. Run setup_windows.bat first.'
}
exit $LASTEXITCODE
