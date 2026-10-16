$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'
$frontend = Join-Path $root 'frontend'

if (-not (Test-Path $python)) {
    throw "Project virtual environment not found: $python"
}
if (-not (Test-Path (Join-Path $frontend 'node_modules'))) {
    throw 'Frontend dependencies are missing. Run npm ci in the frontend directory first.'
}

$backendCommand = "Set-Location -LiteralPath '$root'; & '$python' -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000"
$frontendCommand = "Set-Location -LiteralPath '$frontend'; npm.cmd run dev -- --host 127.0.0.1"
Start-Process -FilePath 'powershell.exe' -ArgumentList @('-NoExit', '-Command', $backendCommand)
Start-Process -FilePath 'powershell.exe' -ArgumentList @('-NoExit', '-Command', $frontendCommand)
Write-Output 'Backend: http://127.0.0.1:8000/docs'
Write-Output 'Frontend: http://127.0.0.1:5173'