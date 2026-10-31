#!/bin/bash
# Торговый терминал: первый запуск создаёт окружение и ставит зависимости.
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -q -e ".[terminal]"
qlab-terminal "$@"
