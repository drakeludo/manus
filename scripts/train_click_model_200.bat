@echo off
setlocal
cd /d "%~dp0.."
python -m pip install --upgrade pip
python -m pip install -r requirements-vds.txt
python scripts\train_click_model.py --data-dir datasets\click_oranges --epochs 200 --batch-size 2 --workers 0 --output-model models\orange_click.pt
python scripts\export_click_onnx.py --checkpoint models\orange_click.pt --output models\orange_click.onnx
