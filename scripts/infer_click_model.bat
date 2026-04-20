@echo off
setlocal
cd /d "%~dp0.."
python scripts\infer_click_model.py --images-dir . --model models\orange_click.onnx --output-dir eval_click_model
