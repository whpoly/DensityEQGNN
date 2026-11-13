param(
    [Parameter(Mandatory = $true)]
    [int]$ProcessId,
    [Parameter(Mandatory = $true)]
    [string]$BenchmarkDir,
    [Parameter(Mandatory = $true)]
    [int]$ExpectedCases
)

$ErrorActionPreference = "Stop"
Wait-Process -Id $ProcessId

$resolvedBenchmark = (Resolve-Path -LiteralPath $BenchmarkDir).Path
$manifestPath = Join-Path $resolvedBenchmark "manifest.json"
$casesPath = Join-Path $resolvedBenchmark "cases.txt"
$manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
$cases = @(
    Get-Content -LiteralPath $casesPath |
        Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
)

if (-not $manifest.cases.charge3net) {
    throw "ChargE3Net generation did not finish successfully"
}
if ($cases.Count -ne $ExpectedCases) {
    throw "Expected $ExpectedCases cases, found $($cases.Count)"
}
foreach ($relativeCase in $cases) {
    $caseDir = Join-Path $resolvedBenchmark $relativeCase
    foreach ($fileName in @("INCAR", "POSCAR", "POTCAR", "KPOINTS")) {
        if (-not (Test-Path -LiteralPath (Join-Path $caseDir $fileName))) {
            throw "Missing $fileName under $caseDir"
        }
    }
}

$parent = Split-Path -Parent $resolvedBenchmark
$name = Split-Path -Leaf $resolvedBenchmark
$archive = Join-Path $parent "$name.tar.gz"
tar -czf $archive -C $parent $name
if ($LASTEXITCODE -ne 0) {
    throw "tar failed with exit code $LASTEXITCODE"
}
Get-Item -LiteralPath $archive |
    Select-Object FullName, Length, LastWriteTime
