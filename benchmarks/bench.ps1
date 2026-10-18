param(
  [Parameter(Mandatory)][string]$Model,
  [string]$Label = "run",
  [string]$Threads = "4",
  [string]$Extra = "-p 128,512 -n 64 -r 3"
)
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
New-Item -ItemType Directory -Force results | Out-Null
$batt = (adb shell dumpsys battery | Select-String '^\s*(USB powered|AC powered|status|level|temperature):' | ForEach-Object { $_.Line.Trim() }) -join ';'
$skin = ((adb shell dumpsys thermalservice | Select-String 'mName=SKIN' | Select-Object -First 1).Line.Trim()) -replace ',',';'
"$ts,$Label,$Model,$batt,$skin" | Add-Content results\meta.csv
adb shell "cd /data/local/tmp/llama && LD_LIBRARY_PATH=. taskset f0 ./llama-bench -m /data/local/tmp/models/$Model -t $Threads $Extra -o csv 2>/dev/null" | Out-File -Encoding utf8 "results\${ts}_$Label.csv"
Write-Host "saved results\${ts}_$Label.csv  [$batt | $skin]"