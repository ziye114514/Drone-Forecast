param(
    [Parameter(Mandatory=$true)][string]$Raw,
    [string]$Output,
    [double]$Start=0,
    [double]$Duration=0,
    [switch]$Preview
)
$ErrorActionPreference='Stop'
$workspace=Split-Path -Parent $PSScriptRoot
$python=Join-Path $workspace '.venv-event311\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Use Python matching your Metavision SDK; see PAPER_REPRODUCTION.md.' }
if (-not $Output) { $Output=Join-Path $workspace ('output\paper_prediction_'+(Get-Date -Format 'yyyyMMdd_HHmmss')) }
$arguments=@((Join-Path $PSScriptRoot 'core\run_paper.py'),'--raw',$Raw,'--output',$Output,'--start',[string]$Start,'--duration',[string]$Duration)
if ($Preview) { $arguments+='--preview' }
& $python @arguments
if ($LASTEXITCODE -ne 0) { throw 'Prediction failed; retain output for diagnostics and choose a new output path for rerun.' }
