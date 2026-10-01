# Publica o IVY LIBRAS na internet (so enquanto esta janela ficar aberta).
# Host Error = tunel no ar sem o Flask na porta 7860. Este script nao solta o link antes do /api/health.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$py = Join-Path $root ".venv\Scripts\python.exe"
$tools = Join-Path $root "tools"
$cf = Join-Path $tools "cloudflared.exe"
$errFile = Join-Path $tools "tunnel.log"
$outFile = Join-Path $tools "tunnel-out.log"

if (-not (Test-Path $py)) {
    throw "Python do projeto nao encontrado. Use a pasta .venv neste computador."
}

New-Item -ItemType Directory -Path $tools -Force | Out-Null

# Tunel anterior deixa o tunnel.log travado. Encerra antes de apagar o log.
Get-Process -Name "cloudflared" -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host "Encerrando tunel antigo (PID $($_.Id))..."
    Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Seconds 1

if (-not (Test-Path $cf)) {
    Write-Host "Baixando o tunel (so na primeira vez)..."
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -Uri "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe" -OutFile $cf -UseBasicParsing
}

$preventSleep = Add-Type -MemberDefinition @"
[DllImport("kernel32.dll")]
public static extern uint SetThreadExecutionState(uint esFlags);
"@ -Name Native -Namespace IvyPower -PassThru
[void]$preventSleep::SetThreadExecutionState(2147483651)

function Test-IvyUp {
    try {
        $r = Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:7860/api/health" -TimeoutSec 2
        return $r.StatusCode -eq 200
    } catch {
        return $false
    }
}

function Start-IvyServer {
    Write-Host "Iniciando o IVY LIBRAS na porta 7860..."
    $arg = "/k title IVY LIBRAS - servidor NAO FECHE && `"$py`" -u app.py"
    return Start-Process -FilePath "cmd.exe" -ArgumentList $arg -WorkingDirectory $root -PassThru
}

$flask = $null
if (-not (Test-IvyUp)) {
    $flask = Start-IvyServer
}
$ok = $false
foreach ($i in 1..50) {
    Start-Sleep -Seconds 1
    if (Test-IvyUp) { $ok = $true; break }
}
if (-not $ok) {
    throw "O totem nao subiu em http://127.0.0.1:7860. Deixe a janela 'IVY LIBRAS - servidor' aberta e rode de novo."
}

foreach ($name in @("tunnel.log", "tunnel-out.log")) {
    $f = Join-Path $tools $name
    if (Test-Path $f) {
        try { Remove-Item $f -Force -ErrorAction Stop } catch { }
    }
}
if (Test-Path $errFile) { $errFile = Join-Path $tools ("tunnel-{0}.log" -f (Get-Date -Format "HHmmss")) }
if (Test-Path $outFile) { $outFile = Join-Path $tools ("tunnel-out-{0}.log" -f (Get-Date -Format "HHmmss")) }

Write-Host "Servidor local OK. Abrindo o tunel..."
$tunnel = Start-Process -FilePath $cf -ArgumentList "tunnel","--url","http://127.0.0.1:7860","--protocol","http2","--no-autoupdate" `
    -RedirectStandardError $errFile -RedirectStandardOutput $outFile -PassThru -WindowStyle Hidden

$url = $null
foreach ($i in 1..45) {
    Start-Sleep -Seconds 1
    foreach ($log in @($errFile, $outFile)) {
        if (-not (Test-Path $log)) { continue }
        $txt = Get-Content $log -Raw -ErrorAction SilentlyContinue
        if ($txt -match "https://[a-z0-9-]+\.trycloudflare\.com") {
            $url = $Matches[0]
            break
        }
    }
    if ($url) { break }
}

if (-not $url) {
    Write-Host "Nao consegui ler o link. Veja o arquivo:"
    Write-Host $errFile
    if (Test-Path $errFile) { Get-Content $errFile }
} else {
    if (-not (Test-IvyUp)) {
        Write-Host "AVISO: o tunel subiu, mas o servidor local caiu. Subindo de novo..."
        if (-not $flask -or $flask.HasExited) { $flask = Start-IvyServer }
        Start-Sleep -Seconds 8
    }
    $linkFile = Join-Path ([Environment]::GetFolderPath("Desktop")) "IVY_LIBRAS_LINK.txt"
    @"
IVY LIBRAS - link da apresentacao

$url

1. Confira NESTE PC: http://127.0.0.1:7860  (tem que abrir o totem)
2. So depois abra o link acima no celular/outro notebook.
3. Se aparecer Host Error, o servidor local caiu: nao feche a janela preta do servidor.

Deixe este computador ligado e as janelas do INICIAR_ONLINE abertas.
"@ | Set-Content -Path $linkFile -Encoding UTF8
    try { Set-Clipboard -Value $url } catch {}
    Write-Host ""
    Write-Host "================================================"
    Write-Host "  1) Teste neste PC:  http://127.0.0.1:7860"
    Write-Host "  2) Link publico:    $url"
    Write-Host "================================================"
    Write-Host "  Salvo em: $linkFile"
    Write-Host "  Nao feche esta janela NEM a janela do servidor."
    Write-Host ""
}

try {
    while ($tunnel -and -not $tunnel.HasExited) {
        if (-not (Test-IvyUp)) {
            Write-Host "$(Get-Date -Format HH:mm:ss) Servidor caiu. Subindo de novo..."
            $flask = Start-IvyServer
        }
        Start-Sleep -Seconds 4
    }
} finally {
    if ($tunnel -and -not $tunnel.HasExited) {
        Stop-Process -Id $tunnel.Id -Force -ErrorAction SilentlyContinue
    }
    if ($flask -and -not $flask.HasExited) {
        Stop-Process -Id $flask.Id -Force -ErrorAction SilentlyContinue
    }
    [void]$preventSleep::SetThreadExecutionState(2147483648)
}
