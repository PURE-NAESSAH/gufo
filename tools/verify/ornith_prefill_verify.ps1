# Ornith-1.5-122B prefill optimization verification (run when GPU is free).
# Usage: powershell -ExecutionPolicy Bypass -File tools\verify\ornith_prefill_verify.ps1
# All runs use `prompt` (native qwen35_moe engine; `bench` lacks qwen35_moe support).
# Logs: build\verify_logs\<timestamp>\*.txt  (~5-10 min warm, longer if page cache cold)
$ErrorActionPreference = 'Continue'
$repo  = 'D:\GIT\gufo'
$new   = Join-Path $repo 'build\windows-release\gufo.exe'
$old   = Join-Path $repo '..\gufo-base\build\windows-release\gufo.exe'
$model = 'D:\AI_Stack\Models\Harsha001\Ornith-1.5-122B-ROCmFPX\Ornith-1.5-122B-ROCmFP4-Q4_0_STRIX-00001-of-00002.gguf'
$fx    = Join-Path $repo 'tools\verify\fixtures'
$log   = Join-Path $repo ('build\verify_logs\' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
New-Item -ItemType Directory -Force -Path $log | Out-Null
$sum = Join-Path $log 'summary.txt'

function Vram-Snapshot {
  try {
    $c = (Get-Counter '\GPU Process Memory(*)\Dedicated Usage' -ErrorAction Stop).CounterSamples |
         Where-Object CookedValue -gt 1GB
    foreach ($s in $c) { Add-Content $sum ("  vram: {0} {1:N1} GB" -f $s.Path, ($s.CookedValue/1GB)) }
  } catch { Add-Content $sum '  vram: counter unavailable' }
}

function Step {
  param([string]$Name, [string]$Exe, [string[]]$CmdArgs)
  Add-Content $sum ''
  Add-Content $sum ("=== [{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $Name)
  if (-not (Test-Path $Exe)) { Add-Content $sum "  SKIPPED: exe missing ($Exe)"; return }
  $out = Join-Path $log "$Name.txt"
  "CMD: $Exe $($CmdArgs -join ' ')" | Out-File -Encoding utf8 $out
  Vram-Snapshot
  $sw = [Diagnostics.Stopwatch]::StartNew()
  & $Exe @CmdArgs *>&1 | Out-File -Encoding utf8 -Append $out
  $code = $LASTEXITCODE
  $sw.Stop()
  Add-Content $sum ("  exit={0} wall={1}s out={2}" -f $code, [math]::Round($sw.Elapsed.TotalSeconds,1), $out)
}

function PromptStep {
  param([string]$Name, [string]$Exe, [string]$Fixture, [string]$N)
  Step $Name $Exe @('prompt','-m',$model,'-f',(Join-Path $fx $Fixture),'-n',$N,'-t','0','-s','42','--think','on','-v')
}

Add-Content $sum "ornith_prefill_verify started $(Get-Date -Format o)"
Add-Content $sum "new=$new`nold=$old`nmodel=$model"
if (-not (Test-Path $model)) { Add-Content $sum "FATAL: model missing"; exit 1 }

Step '00_diagnose' $new @('diagnose')
PromptStep '01_new_timed_512'     $new 'timed_512.txt'  '1'
PromptStep '02_old_timed_512'     $old 'timed_512.txt'  '1'
PromptStep '03_new_timed_2658'    $new 'timed_2658.txt' '1'
PromptStep '04_old_timed_2658'    $old 'timed_2658.txt' '1'
PromptStep '05_new_context_3800'  $new 'timed_3800.txt' '24'
PromptStep '06_old_context_3800'  $old 'timed_3800.txt' '24'
PromptStep '07_new_prompt_en'     $new 'prompt_en.txt'  '96'
PromptStep '08_old_prompt_en'     $old 'prompt_en.txt'  '96'
PromptStep '09_new_prompt_fr'     $new 'prompt_fr.txt'  '96'
PromptStep '10_old_prompt_fr'     $old 'prompt_fr.txt'  '96'

Add-Content $sum ''
Add-Content $sum 'ALL STEPS COMPLETE'
Write-Host "Done. Send me the folder: $log"
