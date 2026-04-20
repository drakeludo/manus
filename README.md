# Orange Bot

Сборка бота для сбора апельсинов в `Majestic RP`.

## Стек

- `CustomOrangeDetector`: ONNX-модель `heatmap + mask`, которая должна заменить `YOLO` как основной нейродетектор.
- `OrangeVision`: агрессивный color detector по экрану.
- `TemplateDetector`: шаблоны для старта мини-игры и добора целей.
- `MainBrain`: объединяет сигналы модели, цвета и шаблонов.
- `Coverage Fill`: добирает непрокрытые цветовые цели, чтобы закрывать всё дерево.
- `MinigameStateTracker`: не даёт кликать вне активной мини-игры.

## Структура

```text
src/orange_bot/    Основной пакет
scripts/           Установка, датасет и обучение
models/            ONNX/PT веса и шаблоны
main.py            Точка входа
```

## Установка

```bat
install.bat
```

Или вручную:

```bat
python -m pip install --upgrade pip
python -m pip install -e .
```

## Запуск

```bat
start.bat
```

Или вручную:

```bat
python main.py
```

## Управление

- `F6` - старт/пауза
- `F7` - выход

## Какие модели ждёт проект

- `models/orange_click.onnx`:
  основная кастомная модель центров апельсинов.
- `models/orange_click_refine.onnx`:
  вторичная refine-модель, если нужна двухмодельная связка.
- `models/orange.png`:
  шаблон апельсина.
- `models/minigame_start.png`:
  шаблон старта мини-игры.

`YOLO` в конфиге оставлен только как legacy fallback и по умолчанию выключен.

## Подготовка bootstrap-датасета

```bat
python scripts/export_yolo_dataset.py --images-dir . --output-dir datasets/majestic_oranges
```

## Подготовка click-датасета под новую модель

```bat
scripts\build_click_dataset.bat
```

Или вручную:

```bat
python scripts/export_click_dataset.py --images-dir . --output-dir datasets/click_oranges
```

## Обучение новой модели

```bat
scripts\train_click_model.bat
```

## Экспорт в ONNX

```bat
scripts\export_click_onnx.bat
```

## Прогон ONNX по скринам

```bat
scripts\infer_click_model.bat
```

## Smoke-train legacy detector

```bat
python scripts/train_custom_yolo.py --data datasets/majestic_oranges/dataset.yaml --epochs 1 --workers 0
```

## Настройка

Параметры лежат в `src/orange_bot/config.py`.
