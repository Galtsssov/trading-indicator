@echo off
rem Trading terminal: first run creates .venv and installs dependencies.
cd /d "%~dp0"
if not exist .venv (
  py -3 -m venv .venv || python -m venv .venv
)
call .venv\Scripts\activate
pip install -q -e ".[terminal]"
qlab-terminal %*
pause
