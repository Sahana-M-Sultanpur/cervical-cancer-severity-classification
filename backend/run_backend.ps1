$ErrorActionPreference = "Stop"

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$venvPython = "C:\Users\sulta\cervical_backend_venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    throw "Backend virtual environment was not found at $venvPython"
}

# Avoid OneDrive permission issues from Python bytecode cache writes.
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTHONIOENCODING = "utf-8"

Set-Location $projectRoot
& $venvPython -m backend.app
