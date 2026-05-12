#requires -version 5
<#
  Phase A: text-injection A/B against adapter_phase4_mixed_format.pt.

  Runs three variants of free_form_transfer_test.py against the SAME
  checkpoint at n=1998, NLI on:

    1. comma_list       (full: color + material + shape)
    2. comma_list (no color)    -- ablates color specifically
    3. numbered         (full, per-object numbered template)

  Baseline (no injection) already exists at
    compsac_2026_code/clevrer_benchmark/results/phase4/phase4_FREEFORM_HELDOUT.json
  so the 4-way comparison is:

    baseline   vs   comma_list   vs   comma_list_no_color   vs   numbered

  Each run takes ~8-12 min on an RTX 5080-class GPU (smoke test clocked
  ~4 q/s). All three runs share the same gen_seed=42 so the sampled
  1998 heldout-valid questions are identical across runs, i.e. per-
  question differences are attributable to the prompt change.

  Results land in:
    compsac_2026_code/clevrer_benchmark/results/phase_a_injection/
  with descriptive filenames so the aggregator can pick them up later.
#>

# Use 'Continue' rather than 'Stop' because we call python (a "native"
# command) and PowerShell's "Stop" converts anything written to stderr
# (even FutureWarnings from third-party packages) into a terminating
# PowerShell error. That kills the pipeline even when python's actual
# exit code is 0. We check $LASTEXITCODE explicitly per-run instead.
$ErrorActionPreference = 'Continue'
$repoRoot   = (Resolve-Path "$PSScriptRoot/../../..").Path
$scriptPath = "compsac_2026_code/clevrer_benchmark/scripts/free_form_transfer_test.py"
$ckpt       = "compsac_2026_code/checkpoints/adapter_phase4_mixed_format.pt"
$resultDir  = "compsac_2026_code/clevrer_benchmark/results/phase_a_injection"

function Invoke-FreeFormRun {
    param(
        [string]$Label,
        [string[]]$PyArgs,
        [string]$OutJson,
        [string]$LogFile
    )
    Write-Host ""
    Write-Host "=== $Label ===" -ForegroundColor Cyan
    Write-Host "  -> $OutJson"
    # Redirect stderr (2) into stdout (1) at the process level so
    # Tee-Object sees the full combined stream without PowerShell
    # error-stream noise getting promoted to NativeCommandError.
    & python $scriptPath @PyArgs 2>&1 | Tee-Object -FilePath $LogFile
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        Write-Host "  [FAIL] $Label exited $code -- see $LogFile" -ForegroundColor Red
        throw "run '$Label' failed with exit $code"
    }
    Write-Host "  [OK] $Label complete" -ForegroundColor Green
}

Push-Location $repoRoot
try {
    New-Item -ItemType Directory -Path $resultDir -Force | Out-Null

    $common = @(
        "--adapter_checkpoint", $ckpt,
        "--n", "1998",
        "--heldout",
        "--seed", "42",
        "--gen_seed", "42",
        "--inject_scene_text"
    )

    $out1 = "$resultDir/inject_COMMALIST_n1998.json"
    Invoke-FreeFormRun -Label "[1/3] comma_list (full)" `
        -PyArgs ($common + @("--scene_text_style", "comma_list", "--out", $out1)) `
        -OutJson $out1 -LogFile "$resultDir/inject_COMMALIST_n1998.log"

    $out2 = "$resultDir/inject_COMMALIST_NOCOLOR_n1998.json"
    Invoke-FreeFormRun -Label "[2/3] comma_list (no color)" `
        -PyArgs ($common + @("--scene_text_style", "comma_list", "--scene_text_no_color", "--out", $out2)) `
        -OutJson $out2 -LogFile "$resultDir/inject_COMMALIST_NOCOLOR_n1998.log"

    $out3 = "$resultDir/inject_NUMBERED_n1998.json"
    Invoke-FreeFormRun -Label "[3/3] numbered" `
        -PyArgs ($common + @("--scene_text_style", "numbered", "--out", $out3)) `
        -OutJson $out3 -LogFile "$resultDir/inject_NUMBERED_n1998.log"

    Write-Host ""
    Write-Host "All three runs finished." -ForegroundColor Green
    Write-Host "  1. $out1"
    Write-Host "  2. $out2"
    Write-Host "  3. $out3"
} finally {
    Pop-Location
}
