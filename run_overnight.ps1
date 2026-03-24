param(
    [int]$FramesPerTown = 70000,
    [int]$TargetRows = 210000,
    [int]$MaxTicksPerEpisode = 3000,
    [double]$MinEpisodeMeanSpeed = 12.0,
    [double]$MaxEpisodeIdleRatio = 0.35,
    [int]$MinEpisodeFrames = 120,
    [switch]$ShowPreview
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
$src = Join-Path $repo "src"
$logsDir = Join-Path $repo "logs"
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null
$logFile = Join-Path $logsDir ("overnight_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")

$env:FRAMES_PER_TOWN = "$FramesPerTown"
$env:TARGET_ROWS = "$TargetRows"
$env:MAX_TICKS_PER_EPISODE = "$MaxTicksPerEpisode"
$env:MIN_EPISODE_MEAN_SPEED = "$MinEpisodeMeanSpeed"
$env:MAX_EPISODE_IDLE_RATIO = "$MaxEpisodeIdleRatio"
$env:MIN_EPISODE_FRAMES = "$MinEpisodeFrames"
if ($ShowPreview.IsPresent) {
    $env:COLLECTOR_SHOW_PREVIEW = "1"
} else {
    $env:COLLECTOR_SHOW_PREVIEW = "0"
}

# Use system Python by default because it contains both CARLA and torch in this setup.
$defaultPython = "C:\Program Files\Python312\python.exe"
if (Test-Path $defaultPython) {
    $env:COLLECTOR_PYTHON = $defaultPython
    $env:TRAIN_PYTHON = $defaultPython
} else {
    $env:COLLECTOR_PYTHON = "python"
    $env:TRAIN_PYTHON = "python"
}

Push-Location $src
try {
    Write-Host "Starting overnight pipeline..."
    Write-Host "Log file: $logFile"
    python overnight_pipeline.py *>&1 | Tee-Object -FilePath $logFile
}
finally {
    Pop-Location
}
