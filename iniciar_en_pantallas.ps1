$ErrorActionPreference = 'Stop'

$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$logsDir = Join-Path $projectDir 'logs'
$launcherLog = Join-Path $logsDir 'startup-launcher.log'

New-Item -ItemType Directory -Force -Path $logsDir | Out-Null

function Write-LauncherLog {
    param([string]$Message)

    $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    Add-Content -Path $launcherLog -Value "[$timestamp] $Message"
}

function Test-PortListening {
    param([int]$Port)

    try {
        return @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop).Count -gt 0
    } catch {
        return $false
    }
}

function Wait-ServerReady {
    param(
        [string]$Url,
        [int]$TimeoutSeconds = 45
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $response = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 3
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
                return $true
            }
        } catch {}
        Start-Sleep -Seconds 1
    }

    return $false
}

function Get-ChromePath {
    $candidates = @(
        "$Env:ProgramFiles\Google\Chrome\Application\chrome.exe",
        "$Env:ProgramFiles(x86)\Google\Chrome\Application\chrome.exe",
        "$Env:LocalAppData\Google\Chrome\Application\chrome.exe"
    )

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    $command = Get-Command chrome.exe -ErrorAction SilentlyContinue
    if ($command -and $command.Source) {
        return $command.Source
    }

    throw 'No encontre chrome.exe en las rutas habituales.'
}

function Get-DisplayTargets {
    Add-Type -AssemblyName System.Windows.Forms

    $screens = [System.Windows.Forms.Screen]::AllScreens
    if (-not $screens -or $screens.Count -eq 0) {
        throw 'Windows no devolvio pantallas disponibles.'
    }

    $sorted = $screens | Sort-Object { $_.Bounds.Width * $_.Bounds.Height } -Descending
    $controlScreen = $sorted | Select-Object -First 1
    $outputScreen = $sorted | Where-Object { $_.DeviceName -ne $controlScreen.DeviceName } | Select-Object -First 1

    if (-not $outputScreen) {
        $outputScreen = $controlScreen
    }

    [pscustomobject]@{
        Control = $controlScreen
        Output = $outputScreen
    }
}

try {
    Write-LauncherLog 'Inicio del launcher.'

    $pythonExe = Join-Path $projectDir '.venv\Scripts\python.exe'
    if (-not (Test-Path $pythonExe)) {
        throw "No encontre el entorno virtual esperado en $pythonExe"
    }

    $serverUrl = 'http://127.0.0.1:8000/'
    $outputUrl = 'http://127.0.0.1:8000/fullscreen2?presentation=1'
    $serverStdout = Join-Path $logsDir 'server.stdout.log'
    $serverStderr = Join-Path $logsDir 'server.stderr.log'

    if (-not (Test-PortListening -Port 8000)) {
        Write-LauncherLog 'Puerto 8000 libre. Lanzando server.py.'
        Start-Process -FilePath $pythonExe `
            -ArgumentList 'server.py' `
            -WorkingDirectory $projectDir `
            -WindowStyle Hidden `
            -RedirectStandardOutput $serverStdout `
            -RedirectStandardError $serverStderr | Out-Null
    } else {
        Write-LauncherLog 'Puerto 8000 ya en escucha. No relanzo el servidor.'
    }

    if (-not (Wait-ServerReady -Url $serverUrl -TimeoutSeconds 45)) {
        throw 'El servidor no respondio a tiempo en http://127.0.0.1:8000/'
    }

    $chromeExe = Get-ChromePath
    $targets = Get-DisplayTargets
    $controlBounds = $targets.Control.Bounds
    $outputBounds = $targets.Output.Bounds

    Write-LauncherLog ("Chrome: {0}" -f $chromeExe)
    Write-LauncherLog ("Control en {0} ({1},{2} {3}x{4})" -f $targets.Control.DeviceName, $controlBounds.X, $controlBounds.Y, $controlBounds.Width, $controlBounds.Height)
    Write-LauncherLog ("Salida en {0} ({1},{2} {3}x{4})" -f $targets.Output.DeviceName, $outputBounds.X, $outputBounds.Y, $outputBounds.Width, $outputBounds.Height)

    $controlArgs = @(
        '--new-window',
        "--window-position=$($controlBounds.X),$($controlBounds.Y)",
        "--window-size=$($controlBounds.Width),$($controlBounds.Height)",
        $serverUrl
    )

    $outputArgs = @(
        "--app=$outputUrl",
        '--start-fullscreen',
        "--window-position=$($outputBounds.X),$($outputBounds.Y)",
        "--window-size=$($outputBounds.Width),$($outputBounds.Height)"
    )

    Start-Process -FilePath $chromeExe -ArgumentList $controlArgs | Out-Null
    Start-Sleep -Milliseconds 900
    Start-Process -FilePath $chromeExe -ArgumentList $outputArgs | Out-Null

    Write-LauncherLog 'Ventanas de Chrome abiertas.'
} catch {
    Write-LauncherLog ("ERROR: {0}" -f $_.Exception.Message)
    throw
}