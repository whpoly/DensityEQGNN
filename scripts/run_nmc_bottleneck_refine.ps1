$ErrorActionPreference = 'Stop'
$taskPython = 'C:\Users\User\.conda\envs\kan-cgcnn-cuda\python.exe'
$taskRoot = 'runs/nmc_bottleneck_2026-10-27'
$taskOutput = "$taskRoot/query_eight_refine"
if (Test-Path -LiteralPath "$taskOutput/fit_complete.json") {
    Get-Content -LiteralPath "$taskOutput/fit_complete.json"
    exit 0
}
$taskArguments = @('-u','-m','scripts.train_nmc_paired_readout','--readout','query',
    '--output',$taskOutput,'--parent',"$taskRoot/query_eight/best.pt",
    '--steps',4000,'--probes',4096,'--lr',0.0001,'--min-lr',0.000001)
if (Test-Path -LiteralPath "$taskOutput/last.pt") { $taskArguments += '--resume' }
@{ phase='query_eight_refine'; status='running'; test_evaluated=$false } | ConvertTo-Json | Set-Content -LiteralPath "$taskRoot/pipeline_status.json"
$ErrorActionPreference = 'Continue'
& $taskPython @taskArguments *> "$taskRoot/query_eight_refine.log"
if ($LASTEXITCODE -ne 0) { throw 'Refinement failed; inspect query_eight_refine.log' }
@{ phase='validation_review'; status='refinement_complete'; test_evaluated=$false } | ConvertTo-Json | Set-Content -LiteralPath "$taskRoot/pipeline_status.json"
Get-Content -LiteralPath "$taskOutput/fit_complete.json"
