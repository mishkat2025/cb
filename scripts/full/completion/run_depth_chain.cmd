@echo off
rem Proposal completion: DepthAnything extraction -> pack -> B2-DepthAnything GroupKFold-5 -> LOSO, as a hidden job:
rem     explorer.exe D:\cb\cvdc-physio\scripts\full\completion\run_depth_chain.cmd
rem This window only starts the job and closes. Log outputs\full\results\completion\depth_chain.log, status
rem depth_chain_status.json next to it. Resumable: start it again to continue after a stop.
cd /d "%~dp0..\..\.."
"%~dp0..\..\..\..\.venv\Scripts\python.exe" scripts\full\completion\launch_depth_chain.py
