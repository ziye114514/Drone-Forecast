param(
    [Parameter(Mandatory=$true)][string]$Raw,
    [string]$Output = (Join-Path $PSScriptRoot '..\output\raw_only_project_integrated\custom_run'),
    [double]$Start = 0,
    [double]$Duration = 0
)
$ErrorActionPreference = 'Stop'
$pythonForEvents = Join-Path $PSScriptRoot '..\.venv-event311\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonForEvents)) {
    throw 'Install Python dependencies in the SDK-compatible environment described in README.md.'
}
& $pythonForEvents (Join-Path $PSScriptRoot 'core\run.py') --raw $Raw --output $Output --start $Start --duration $Duration --preview
if ($LASTEXITCODE -ne 0) { throw "Prediction failed with exit code $LASTEXITCODE" }
