# Tunnel local OcuSmart API (port 8000) to the internet.
# Prereq: API running, e.g.  python -m uvicorn main:app --host 127.0.0.1 --port 8000
# Auth (once): https://dashboard.ngrok.com/get-started/your-authtoken
#   ngrok config add-authtoken <YOUR_TOKEN>

$ErrorActionPreference = "Stop"
$port = if ($env:PORT) { $env.PORT } else { 8000 }

Write-Host "Starting ngrok -> http://127.0.0.1:$port" -ForegroundColor Cyan
Write-Host "Inspector (if 4040 busy, ngrok picks 4041/4042): http://127.0.0.1:4040" -ForegroundColor DarkGray
ngrok http $port
