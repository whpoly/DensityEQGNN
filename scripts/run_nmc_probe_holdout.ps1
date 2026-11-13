$ErrorActionPreference = 'Stop'
$taskPython = 'C:\Users\User\.conda\envs\kan-cgcnn-cuda\python.exe'
$taskRoot = 'runs/nmc_probe_holdout_2026-10-26'
$taskRuns = @(
    @{ Name = 'deepdft_one'; Model = 'deepdft'; Samples = @(1); Steps = 4000 },
    @{ Name = 'r16_one'; Model = 'r16'; Samples = @(1); Steps = 4000 },
    @{ Name = 'deepdft_eight'; Model = 'deepdft'; Samples = @(0,1,2,3,4,5,6,7); Steps = 16000 },
    @{ Name = 'r16_eight'; Model = 'r16'; Samples = @(0,1,2,3,4,5,6,7); Steps = 16000 }
)
foreach ($taskRun in $taskRuns) {
    $taskOutput = Join-Path $taskRoot $taskRun.Name
    if (Test-Path -LiteralPath (Join-Path $taskOutput 'result.json')) {
        Write-Output "Already completed: $($taskRun.Name)"
        continue
    }
    $taskArguments = @('-u','-m','scripts.train_nmc_probe_holdout','--model',$taskRun.Model,
        '--samples') + $taskRun.Samples + @('--output-dir',$taskOutput,'--steps',$taskRun.Steps,
        '--probes',1024,'--eval-every',500,'--chunk',2048,'--seed',42,'--lr',0.001,'--min-lr',0.00001,'--loss','l1')
    if (Test-Path -LiteralPath (Join-Path $taskOutput 'last.pt')) { $taskArguments += '--resume' }
    @{ phase=$taskRun.Name; status='running'; steps=$taskRun.Steps } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskRoot 'pipeline_status.json')
    Write-Output "Starting $($taskRun.Name)"
    & $taskPython @taskArguments *> (Join-Path $taskRoot "$($taskRun.Name).log")
    if ($LASTEXITCODE -ne 0) {
        @{ phase=$taskRun.Name; status='failed'; exit_code=$LASTEXITCODE } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskRoot 'pipeline_status.json')
        Get-Content -LiteralPath (Join-Path $taskRoot "$($taskRun.Name).log") -Tail 20
        exit $LASTEXITCODE
    }
    Get-Content -LiteralPath (Join-Path $taskOutput 'result.json') -TotalCount 5
}
@{ status='completed' } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskRoot 'pipeline_status.json')
