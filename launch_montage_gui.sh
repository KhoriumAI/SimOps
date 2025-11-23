#!/bin/bash
# Launcher script for Montage GUI
# This sets the correct Qt plugin path before launching

export QT_PLUGIN_PATH="/opt/anaconda3/lib/python3.13/site-packages/PyQt6/Qt6/plugins"
cd "$(dirname "$0")"
python3 montage_gui.py
