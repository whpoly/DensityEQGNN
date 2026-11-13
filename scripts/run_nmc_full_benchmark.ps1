param(
    [string]$Python = "C:\Users\User\.conda\envs\kan-cgcnn-cuda\python.exe",
    [string]$DataRoot = "D:\density-eqgnn-data",
    [string]$Workspace = "C:\Users\User\Desktop\density EQGNN"
)

$ErrorActionPreference = "Stop"
$env:DENSITY_EQGNN_DATA_ROOT = $DataRoot
Set-Location -LiteralPath $Workspace

function Invoke-Checked {
    param([string[]]$CommandArgs)
    & $Python @CommandArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE`: $CommandArgs"
    }
}

Invoke-Checked @(
    "scripts/prepare_nmc_full_benchmark.py",
    "--raw-dir", (Join-Path $DataRoot "nmc_raw"),
    "--output", (Join-Path $DataRoot "nmc_full_probes")
)

Invoke-Checked @(
    "-m", "density_eqgnn.training.train_compact_coefficients",
    "--config", "configs/nmc_full_l2_to_l4_mace_pbc.yaml"
)

Invoke-Checked @(
    "scripts/evaluate_nmc_full_grid.py",
    "--config", "configs/nmc_full_l2_to_l4_mace_pbc.yaml",
    "--checkpoint", "runs/nmc_full_l2_to_l4_mace_pbc/best.pt",
    "--raw-dir", (Join-Path $DataRoot "nmc_raw"),
    "--split-file", (Join-Path $DataRoot "nmc_full_probes/test.txt"),
    "--output", "runs/nmc_full_l2_to_l4_mace_pbc/test_full_grid_metrics.json"
)

Invoke-Checked @(
    "scripts/summarize_nmc_benchmark.py",
    "--metrics", "runs/nmc_full_l2_to_l4_mace_pbc/test_full_grid_metrics.json",
    "--charge3net-checkpoint",
    (Join-Path $DataRoot "charge3net-reference/models/charge3net_nmc.pt"),
    "--output", "reports/nmc_full_l2_to_l4_pbc_vs_charge3net.md"
)
