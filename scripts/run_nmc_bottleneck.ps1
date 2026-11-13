param([int]$WaitForOracleProcess = 0)
$ErrorActionPreference = 'Stop'
$taskPython = 'C:\Users\User\.conda\envs\kan-cgcnn-cuda\python.exe'
$taskRoot = 'runs/nmc_bottleneck_2026-10-27'
function Invoke-TaskPython {
    param([object[]]$TaskArguments, [string]$TaskLog)
    # Windows PowerShell classifies native stderr warnings as ErrorRecords.
    # Python's exit code determines failure; keep stderr in the diagnostic log.
    $ErrorActionPreference = 'Continue'
    & $taskPython @TaskArguments *> $TaskLog
    if ($LASTEXITCODE -ne 0) { throw "Python failed; inspect $TaskLog" }
}
if ($WaitForOracleProcess -gt 0) {
    $taskExisting = Get-Process -Id $WaitForOracleProcess -ErrorAction SilentlyContinue
    if ($taskExisting) { Wait-Process -Id $WaitForOracleProcess }
}
@{ phase='oracle'; status='running' } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskRoot 'pipeline_status.json')
Invoke-TaskPython -TaskArguments @('-u','-m','scripts.fit_nmc_holdout_oracle') -TaskLog (Join-Path $taskRoot 'oracle_pipeline.log')
$taskOutput = Join-Path $taskRoot 'query_eight'
if (-not (Test-Path -LiteralPath (Join-Path $taskOutput 'fit_complete.json'))) {
    @{ phase='query_eight'; status='running'; steps=16000 } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskRoot 'pipeline_status.json')
    $taskArguments = @('-u','-m','scripts.train_nmc_paired_readout','--readout','query',
        '--output',$taskOutput,'--steps',16000,'--probes',1024,'--eval-every',500,
        '--chunk',2048,'--seed',42,'--lr',0.001,'--min-lr',0.00001,'--query-radial-hidden',25)
    if (Test-Path -LiteralPath (Join-Path $taskOutput 'last.pt')) { $taskArguments += '--resume' }
    Invoke-TaskPython -TaskArguments $taskArguments -TaskLog (Join-Path $taskRoot 'query_eight.log')
}
Invoke-TaskPython -TaskArguments @('-m','scripts.audit_nmc_paired_control') -TaskLog (Join-Path $taskRoot 'paired_control_audit.log')
@{ phase='validation_review'; status='training_complete'; test_evaluated=$false } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskRoot 'pipeline_status.json')
Get-Content -LiteralPath (Join-Path $taskOutput 'fit_complete.json')
