param(
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

function Find-Python {
    $venvPython = Join-Path $Root ".venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        return $venvPython
    }

    $commands = @(
        @("python", @()),
        @("py", @("-3"))
    )
    foreach ($candidate in $commands) {
        $exe = $candidate[0]
        $args = $candidate[1]
        try {
            & $exe @args -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0) {
                return @{ Exe = $exe; Args = $args }
            }
        } catch {
        }
    }
    throw "Python was not found. Install Python 3.10+ or create .venv in the project root."
}

$venvPython = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    $python = Find-Python
    if ($python -is [string]) {
        & $python -m venv ".venv"
    } else {
        & $python.Exe @($python.Args) -m venv ".venv"
    }
}

if (-not $SkipInstall) {
    & $venvPython -m pip install --upgrade pip
    & $venvPython -m pip install -r requirements.txt
}

$addData = "med_audio_lab\utils\wavelet_scales.txt;med_audio_lab\utils"
& $venvPython -m PyInstaller `
    --noconfirm `
    --clean `
    --onefile `
    --windowed `
    --name "MedAudioLab" `
    --add-data $addData `
    --hidden-import "serial.tools.list_ports" `
    "run.py"

Write-Host ""
Write-Host "Done: $Root\dist\MedAudioLab.exe"
