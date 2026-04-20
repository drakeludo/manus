@echo off
setlocal
cd /d "%~dp0.."
python scripts\train_click_model.py --data-dir datasets\click_oranges --epochs 20 --batch-size 4 --workers 0 --output-model models\orange_click.pt
