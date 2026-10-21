@echo off
rem LOSO smoke test (2 subjects per exercise, 2 epochs), detached like launch.cmd:
rem     explorer.exe D:\cb\cvdc-physio\scripts\loso\launch_smoke.cmd
cd /d "%~dp0..\.."
"%~dp0..\..\..\.venv\Scripts\python.exe" scripts\loso\launch.py --smoke %*
