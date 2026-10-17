<#
.SYNOPSIS
    Builds dist\MedAudioLab.exe (MedAudioLab v2) with PyInstaller.

.DESCRIPTION
    1. Creates the virtual environment (.venv) if it does not exist.
    2. Installs requirements.txt and PyInstaller into it (skip with -SkipInstall).
    3. Runs: pyinstaller medaudiolab.spec --noconfirm --clean
    4. Verifies that the executable exists and is not empty.

    The exit code is 0 on success and 1 on any failure.

.PARAMETER SkipInstall
    Do not touch the installed packages (the environment must already contain the
    dependencies and PyInstaller).

.PARAMETER VenvPath
    Virtual environment directory (default: .venv). Relative paths are resolved
    against the project root.

.PARAMETER DistPath
    Output directory for MedAudioLab.exe (default: dist).

.PARAMETER WorkPath
    PyInstaller's scratch directory (default: build).
#>
[CmdletBinding()]
param(
    [switch]$SkipInstall,
    [string]$VenvPath = ".venv",
    [string]$DistPath = "dist",
    [string]$WorkPath = "build"
)

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

$SpecFile = "medaudiolab.spec"
$ExeName = "MedAudioLab.exe"
$MinimumPython = [version]"3.10"
$RecommendedPython = [version]"3.11"
$PyInstallerRequirement = "pyinstaller>=6.3"
$MinimumExeBytes = 1MB


function Resolve-UnderRoot([string]$Path) {
    if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
    return (Join-Path $Root $Path)
}

# Windows PowerShell 5.1 does not stop on a failing native command: check the exit code.
function Invoke-Checked([string]$Description, [string]$Exe, [string[]]$Arguments) {
    Write-Host ""
    Write-Host "==> $Description" -ForegroundColor Cyan
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed (exit code $LASTEXITCODE)."
    }
}

# Returns @{ Exe = ...; Args = ...; Version = ... } for the first Python >= 3.10 found.
function Find-Python {
    $candidates = @(
        @{ Exe = "py"; Args = @("-3") },
        @{ Exe = "python"; Args = @() }
    )
    foreach ($candidate in $candidates) {
        try {
            $text = & $candidate.Exe @($candidate.Args) -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
            if ($LASTEXITCODE -ne 0 -or -not $text) { continue }
            $version = [version]($text | Select-Object -First 1).Trim()
            if ($version -ge $MinimumPython) {
                return @{ Exe = $candidate.Exe; Args = $candidate.Args; Version = $version }
            }
        } catch {
            continue  # command not found, or the Microsoft Store stub
        }
    }
    throw "Python $MinimumPython or newer was not found. Install it from python.org and make sure 'py' or 'python' is on PATH."
}


try {
    $VenvDir = Resolve-UnderRoot $VenvPath
    $DistDir = Resolve-UnderRoot $DistPath
    $WorkDir = Resolve-UnderRoot $WorkPath
    $VenvPython = Join-Path $VenvDir "Scripts\python.exe"
    $ExePath = Join-Path $DistDir $ExeName

    if (-not (Test-Path (Join-Path $Root $SpecFile))) {
        throw "$SpecFile was not found in $Root."
    }

    # --- 1. virtual environment
    if (Test-Path $VenvPython) {
        Write-Host "Using the existing environment: $VenvDir"
    } else {
        $python = Find-Python
        Write-Host "Creating the environment $VenvDir with Python $($python.Version)"
        if ($python.Version -lt $RecommendedPython) {
            Write-Warning "pyproject.toml asks for Python $RecommendedPython or newer; $($python.Version) was found."
        }
        Invoke-Checked "Creating the virtual environment" $python.Exe (@($python.Args) + @("-m", "venv", $VenvDir))
    }

    # --- 2. dependencies
    if ($SkipInstall) {
        Write-Host "Skipping the installation of packages (-SkipInstall)."
        # (a failing probe writes to stderr, which Windows PowerShell 5.1 would turn into an
        # exception under "Stop"; relax the preference just for this check)
        $previousPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & $VenvPython -m PyInstaller --version *> $null
        $pyInstallerFound = ($LASTEXITCODE -eq 0)
        $ErrorActionPreference = $previousPreference
        if (-not $pyInstallerFound) {
            throw "PyInstaller is not installed in $VenvDir. Run the script without -SkipInstall."
        }
    } else {
        Invoke-Checked "Upgrading pip" $VenvPython @("-m", "pip", "install", "--upgrade", "pip")
        Invoke-Checked "Installing requirements.txt" $VenvPython @("-m", "pip", "install", "-r", (Join-Path $Root "requirements.txt"))
        Invoke-Checked "Installing PyInstaller" $VenvPython @("-m", "pip", "install", $PyInstallerRequirement)
    }

    # --- 3. build (a stale executable must never pass the check below)
    if (Test-Path $ExePath) {
        Remove-Item $ExePath -Force
    }
    Invoke-Checked "Running PyInstaller" $VenvPython @(
        "-m", "PyInstaller", $SpecFile,
        "--noconfirm", "--clean",
        "--distpath", $DistDir,
        "--workpath", $WorkDir
    )

    # --- 4. verify
    if (-not (Test-Path $ExePath)) {
        throw "PyInstaller finished but $ExePath does not exist."
    }
    $size = (Get-Item $ExePath).Length
    if ($size -lt $MinimumExeBytes) {
        throw "$ExePath exists but is only $size bytes - the build is incomplete."
    }

    Write-Host ""
    Write-Host ("Done: {0} ({1:N1} MB)" -f $ExePath, ($size / 1MB)) -ForegroundColor Green
    exit 0
} catch {
    Write-Host ""
    Write-Host "BUILD FAILED: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
