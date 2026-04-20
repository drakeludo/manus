@echo off
setlocal
cd /d "%~dp0.."
python scripts\export_click_onnx.py --checkpoint models\orange_click.pt --output models\orange_click.onnx
