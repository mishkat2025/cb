@echo off
rem Start (or resume) the LOSO job detached from every terminal and app:
rem     explorer.exe D:\cb\cvdc-physio\scripts\loso\launch.cmd     (or double-click this file)
rem Log runs\full\perex_loso\train.log; status: python scripts\loso\status.py; stop: scripts\loso\stop.ps1 (README.md).
cd /d "%~dp0..\.."
"%~dp0..\..\..\.venv\Scripts\python.exe" scripts\loso\launch.py %*
