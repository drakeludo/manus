@echo off
setlocal
cd /d "%~dp0.."
python scripts\export_click_dataset.py --images-dir . --output-dir datasets\click_oranges
