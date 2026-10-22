# Stop the LOSO job (every python / cmd process whose command line runs cvdc.train.run_loso). The run in progress has
# no done.json, so the next launch redoes it; finished runs are kept.
# usage:  powershell -NoProfile -ExecutionPolicy Bypass -File D:\cb\cvdc-physio\scripts\full\loso\stop.ps1
$p = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'cvdc\.train\.run_loso' -and $_.ProcessId -ne $PID }
if (-not $p) { Write-Output "no LOSO job running"; exit 0 }
$p | ForEach-Object {
    Write-Output ("stopping pid {0} ({1})" -f $_.ProcessId, $_.Name)
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
}
