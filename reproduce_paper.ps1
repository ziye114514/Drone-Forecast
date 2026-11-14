$ErrorActionPreference='Stop'
$workspace=Split-Path -Parent $PSScriptRoot
$python=Join-Path $workspace '.venv-event311\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'See PAPER_REPRODUCTION.md for compatible Metavision SDK and Python setup.' }
function Invoke-PaperTool([string]$Name,[string[]]$ToolArguments) {
    & $python (Join-Path $PSScriptRoot ('tools\'+$Name)) @ToolArguments
    if ($LASTEXITCODE -ne 0) { throw "Failed: $Name" }
}
# Existing frozen runs are reused. Never replace the freeze or published logs.
$out=Join-Path $workspace 'output\paper_reproduction_v030'
if (-not (Test-Path -LiteralPath (Join-Path $out 'experiment_protocol.json'))) { Invoke-PaperTool 'paper_experiment.py' @('freeze') }
& $python -m unittest discover -s (Join-Path $PSScriptRoot 'tests') -v 2>&1 | Tee-Object -FilePath (Join-Path $out 'unit_tests.txt')
if ($LASTEXITCODE -ne 0) { throw 'Unit tests failed.' }
Invoke-PaperTool 'paper_experiment.py' @('run')
Invoke-PaperTool 'paper_experiment.py' @('evaluate')
Invoke-PaperTool 'paper_experiment.py' @('prefix')
Invoke-PaperTool 'paper_controlled_flight.py' @()
# For the optional public benchmark see RELATED_WORK_AUDIT.md; it needs the
# separately obtained original EEPPR sources/data and CUDA Python environment.
# This local reproduction script reuses the public benchmark evidence included
# in the delivery archive. It does not silently download third-party data.
Invoke-PaperTool 'paper_timing.py' @()
Invoke-PaperTool 'paper_visuals.py' @('--run',(Join-Path $out 'runs\train6'),'--output',(Join-Path $out 'visuals\four_rotors'),'--start','22','--duration','6')
Invoke-PaperTool 'paper_visuals.py' @('--run',(Join-Path $out 'runs\train6'),'--output',(Join-Path $out 'visuals\side_view'),'--start','109','--duration','5')
Invoke-PaperTool 'build_paper_report.py' @()
Invoke-PaperTool 'validate_paper_delivery.py' @()
Invoke-PaperTool 'package_paper.py' @()
