$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
$workspaceRoot=Split-Path -Parent $projectRoot
$buildOutput=Join-Path $workspaceRoot 'output\geometry206_20261108'
New-Item -ItemType Directory -Force -Path $buildOutput | Out-Null
$vsDev='C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat'
$nativeCpp=Join-Path $projectRoot 'native\geometry_kernels206.cpp'
$buildCmd=Join-Path $buildOutput 'build.cmd'
$commandText=@"
@echo off
call "$vsDev" -arch=x64 -host_arch=x64 >nul
if errorlevel 1 exit /b 1
cl /nologo /std:c++17 /EHsc /O2 /fp:strict /arch:AVX2 /openmp /MD /LD /I"D:\Prophesee\third_party\include" "$nativeCpp" /Fo"$buildOutput\ellipse.obj" /Fe"$buildOutput\geometry_kernels.dll" /link /LIBPATH:"D:\Prophesee\third_party\lib" opencv_core4.lib opencv_imgproc4.lib
exit /b %errorlevel%
"@
[System.IO.File]::WriteAllText($buildCmd,$commandText,[System.Text.Encoding]::Default)
Push-Location $buildOutput
try { & $env:ComSpec /d /c $buildCmd; if ($LASTEXITCODE -ne 0) { throw 'AMS batch compile failed' } }
finally { Pop-Location }
