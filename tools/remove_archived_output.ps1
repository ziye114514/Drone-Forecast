# Delete only selections already verified in the complete history archive.
$ErrorActionPreference = 'Stop'
$workspaceRoot = [IO.Path]::GetFullPath('D:\drone forecast')
$outputRoot = [IO.Path]::GetFullPath((Join-Path $workspaceRoot 'output')).TrimEnd('\')
$recordRoot = Join-Path $outputRoot 'cleanup_20261028'
$proof = Get-Content -LiteralPath (Join-Path $recordRoot 'archive_verified.json') -Raw -Encoding utf8 | ConvertFrom-Json
if (-not $proof.all_files_verified) { throw 'Archive verification is incomplete.' }
$archivePath = [IO.Path]::GetFullPath([string]$proof.archive)
$expectedArchive = Join-Path $workspaceRoot 'archive\output_history_20261028.zip'
if (-not [string]::Equals($archivePath, $expectedArchive, [StringComparison]::OrdinalIgnoreCase)) { throw 'Unexpected archive target.' }
$archiveItem = Get-Item -LiteralPath $archivePath
if ($archiveItem.PSIsContainer -or $archiveItem.Length -ne $proof.archive_bytes) { throw 'Archive size mismatch.' }
$protected = @('checkpoints','github_release_final_20261027','github_release_update_20261028','shared_template_budget_20261027','three_continuation_20261026','shared_angular_phase_20261027','cleanup_20261028','结果入口.md','查看实验结果.html')
$targets = @()
foreach ($relative in $proof.intended_delete_entries) {
    $normalized = ([string]$relative).Replace('/', '\')
    $target = [IO.Path]::GetFullPath((Join-Path $outputRoot $normalized))
    if (-not $target.StartsWith($outputRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw "Outside output: $target" }
    if ([string]::Equals($target, $outputRoot, [StringComparison]::OrdinalIgnoreCase)) { throw 'Refuse output root.' }
    $top = ($normalized -split '\\')[0]
    if (($protected -contains $top) -or $top.StartsWith('dense_labels_')) {
        $allowVerification = $normalized -match '^github_release_(?:final_20261027|update_20261028)\\verification\\[^\\]+$'
        if (-not $allowVerification) { throw "Protected retained target: $relative" }
    }
    $item = Get-Item -LiteralPath $target -Force
    if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw "Refuse reparse point: $target" }
    $targets += [PSCustomObject]@{ Relative = [string]$relative; Path = $target; Directory = $item.PSIsContainer }
}
# All absolute paths are checked before the first recursive delete.
$deleted = [Collections.Generic.List[string]]::new()
foreach ($entry in $targets) {
    if ($entry.Directory) { Remove-Item -LiteralPath $entry.Path -Recurse -Force }
    else { Remove-Item -LiteralPath $entry.Path -Force }
    $deleted.Add($entry.Relative)
}
$result = @{ completed = $true; archiveVerified = $true; deleted = @($deleted.ToArray()); retainedHistoryArchive = $archivePath; utc = [DateTime]::UtcNow.ToString('o') }
[IO.File]::WriteAllText((Join-Path $recordRoot 'deletion.json'), ($result | ConvertTo-Json -Depth 5), [Text.UTF8Encoding]::new($false))
Write-Output ("Archived selections removed: " + $deleted.Count)
