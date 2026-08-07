# =============================================================================
#  Upload raw data + checkpoints from Windows to VSC project storage.
#
#      .\scripts\upload_to_vsc.ps1 -User vsc38338
#      .\scripts\upload_to_vsc.ps1 -User vsc38338 -DryRun
#
#  WHY THIS EXISTS: `rsync` does not exist on Windows. Git Bash ships `ssh` and
#  `scp` but not `rsync`, and PowerShell has neither. `scp -r` is present on every
#  Windows 10/11 (OpenSSH client is installed by default) and is all we need — the
#  raw datasets are copied once and then live on staging.
#
#  scp has no `--progress` and no resume. For a handful of GB that is fine. If a
#  transfer is interrupted, re-run: `scp -r` overwrites rather than duplicating.
# =============================================================================

param(
    [Parameter(Mandatory = $true)][string]$User,
    [string]$VscHost = "login.hpc.kuleuven.be",
    [string]$Staging = "/lustre1/project/stg_00211/CreditICL",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$remote = "$User@$VscHost"

Write-Host "repo   : $repo"
Write-Host "remote : ${remote}:$Staging"
Write-Host ""

# The directories that must exist before scp can write into them. scp will not
# create intermediate directories, so this runs first.
$mkdir = "mkdir -p '$Staging/data/raw' '$Staging/data/processed' '$Staging/checkpoints' '$Staging/prior_cache'"
Write-Host ">>> creating the directory tree on VSC"
if ($DryRun) {
    Write-Host "    [dry run] ssh $remote `"$mkdir`""
} else {
    ssh $remote $mkdir
    if ($LASTEXITCODE -ne 0) { throw "ssh failed - check your VSC account and any MFA prompt" }
}

# What to upload, and what deliberately not to.
#   data/raw     - IRREPLACEABLE. The datasets themselves.
#   checkpoints  - TabPFN/TabICL weights, needed by the eval pipeline.
# NOT uploaded:
#   prior_cache  - generated ON the cluster; 4-5 GB per variant, pointless to ship
#   data/processed - rebuilt from raw in minutes by scripts/preprocess.py
#   the venv     - VSC needs its own per-architecture build
$targets = @(
    @{ Local = "data\raw"; Remote = "$Staging/data/"; Label = "raw datasets (irreplaceable)" },
    @{ Local = "checkpoints"; Remote = "$Staging/"; Label = "model weights" }
)

foreach ($t in $targets) {
    $src = Join-Path $repo $t.Local
    if (-not (Test-Path $src)) {
        Write-Host ">>> SKIP $($t.Label) - $src does not exist"
        continue
    }
    $size = "{0:N1} MB" -f ((Get-ChildItem $src -Recurse -File | Measure-Object -Property Length -Sum).Sum / 1MB)
    Write-Host ">>> $($t.Label)  ($size)"
    if ($DryRun) {
        Write-Host "    [dry run] scp -r `"$src`" ${remote}:$($t.Remote)"
    } else {
        scp -r "$src" "${remote}:$($t.Remote)"
        if ($LASTEXITCODE -ne 0) { throw "scp failed for $($t.Label)" }
    }
}

Write-Host ""
Write-Host "Done. Verify on VSC with:"
Write-Host "    ssh $remote `"du -sh $Staging/*; ls $Staging/data/raw/lgd | head`""
