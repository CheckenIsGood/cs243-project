@echo off
rem Usage: build.cmd          (build once)
rem        build.cmd watch    (rebuild on every save)
if /I "%~1"=="watch" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0build.ps1" -Watch
) else (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0build.ps1"
)
