param(
    [string]$Output = (Join-Path $PSScriptRoot '..\output\raw_only_project_integrated'),
    [switch]$SkipMedia
)
$ErrorActionPreference = 'Stop'
$eventPython = Join-Path $PSScriptRoot '..\.venv-event311\Scripts\python.exe'
& $eventPython (Join-Path $PSScriptRoot 'tools\run_suite.py') --output $Output --skip-complete
if ($LASTEXITCODE -ne 0) { throw "Inference/evaluation failed with exit code $LASTEXITCODE" }
$finishArguments = @((Join-Path $PSScriptRoot 'tools\finish_integration.py'), '--output', $Output)
if ($SkipMedia) { $finishArguments += '--skip-media' }
& $eventPython @finishArguments
if ($LASTEXITCODE -ne 0) { throw "Report/validation failed with exit code $LASTEXITCODE" }
