# Compile only our C ABI; author headers and installed dependencies stay external.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$workspaceRoot = Split-Path -Parent $projectRoot
$bridgeOutput = Join-Path $workspaceRoot 'output\helix_bridge180_20261107'
New-Item -ItemType Directory -Force -Path $bridgeOutput | Out-Null
$vsDev = 'C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat'
$authorRoot = Join-Path $workspaceRoot 'helixtrack-main'
$nativeCpp = Join-Path $projectRoot 'native\helixtrack_bridge.cpp'
$bridgeDll = Join-Path $bridgeOutput 'helixtrack_bridge.dll'
$objectFile = Join-Path $bridgeOutput 'bridge.obj'
$buildCmd = Join-Path $bridgeOutput 'build.cmd'
$commandText = @"
@echo off
call "$vsDev" -arch=x64 -host_arch=x64 >nul
if errorlevel 1 exit /b 1
cl /nologo /std:c++17 /EHsc /O2 /MD /LD /D_USE_MATH_DEFINES /DNOMINMAX /I"$authorRoot" /I"D:\Prophesee\third_party\include" /I"D:\Prophesee\third_party\include\eigen3" "$nativeCpp" /Fo"$objectFile" /Fe"$bridgeDll" /link /LIBPATH:"D:\Prophesee\third_party\lib" opencv_core4.lib opencv_imgproc4.lib
exit /b %errorlevel%
"@
[System.IO.File]::WriteAllText($buildCmd, $commandText, [System.Text.Encoding]::Default)
Push-Location $bridgeOutput
try { & $env:ComSpec /d /c $buildCmd; if ($LASTEXITCODE -ne 0) { throw 'Bridge compile failed' } }
finally { Pop-Location }
