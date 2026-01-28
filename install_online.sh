#!/usr/bin/env bash
# SimOps Online Installer — Mac/Linux launcher.
# Runs the cross-platform Python installer. Ensure Python 3.8+ is installed.

set -e
cd "$(dirname "$0")"

if ! command -v python3 &>/dev/null; then
    echo "[ERROR] python3 not found. Install Python 3.8+ (e.g. https://www.python.org/downloads/ or brew install python)."
    exit 1
fi

python3 install_online.py
exit $?
