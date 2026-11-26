param(
    [string]$LocalIp = "10.8.30.29",
    [string]$BoardIp = "10.8.30.54",
    [int]$Port = 8080,
    [string]$Ffmpeg = "D:\ffmpeg\bin\ffmpeg.exe",
    [switch]$AutoStart,
    [string]$AutoStartVideo = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Arguments = @(
    "$Root\stream_lab_server.py",
    "--local-ip", $LocalIp,
    "--board-ip", $BoardIp,
    "--ffmpeg", $Ffmpeg,
    "--port", $Port
)
if ($AutoStart) { $Arguments += "--auto-start" }
if ($AutoStartVideo) { $Arguments += @("--auto-start-video", $AutoStartVideo) }
python @Arguments
