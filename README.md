# Orange Bot for Majestic RP

Бот для сбора апельсинов в `Majestic RP`.

Проект состоит из двух частей:
- runtime-бот, который анализирует экран и кликает по найденным целям;
- ML-пайплайн для обучения своей модели вместо `YOLO`.

## Что сейчас делает проект

Во время работы бот:
- нажимает `E`, чтобы активировать дерево;
- проверяет, что мини-игра действительно началась;
- ищет апельсины на экране;
- объединяет сигналы от кастомной модели, цветового детектора и шаблонов;
- выбирает точки клика;
- добирает непрокрытые цветовые цели через `coverage fill`;
- после цикла выходит из триггера и возвращается обратно.

## Главная идея

Основной стек в проекте такой:
- `CustomOrangeDetector`
  ONNX-модель с выходами `center_heatmap + orange_mask`.
- `OrangeVision`
  быстрый color detector по экрану.
- `TemplateDetector`
  шаблоны для старта мини-игры и дополнительного подтверждения.
- `MainBrain`
  сливает сигналы от модели, цвета и шаблонов.
- `Coverage Fill`
  закрывает непрокрытые цели, если модель что-то пропустила.
- `MinigameStateTracker`
  запрещает клик вне активной мини-игры.

`YOLO` оставлен только как legacy fallback. Основной путь теперь не через `YOLO`.

## Структура репозитория

```text
src/orange_bot/
  bot.py                 основной runtime-цикл
  brain.py               логика выбора целей
  click_model.py         новая модель heatmap + mask
  custom_detector.py     ONNX inference для новой модели
  vision.py              цветовой detector
  templates.py           template matching
  state.py               определение состояния мини-игры
  win32_input.py         ввод в Windows
  config.py              все настройки

scripts/
  install.bat
  start.bat
  build_click_dataset.bat
  train_click_model.bat
  train_click_model_200.bat
  export_click_onnx.bat
  infer_click_model.bat
  export_click_dataset.py
  train_click_model.py
  export_click_onnx.py
  infer_click_model.py

models/
  orange_click.pt        checkpoint новой модели
  orange_click.onnx      экспортированная ONNX-модель
  orange.png             шаблон апельсина
  minigame_start.png     шаблон старта мини-игры
```

## Установка

Вариант через батник:

```bat
install.bat
```

Ручная установка:

```bat
python -m pip install --upgrade pip
python -m pip install -e .
```

## Запуск бота

Через батник:

```bat
start.bat
```

Или вручную:

```bat
python main.py
```

## Управление

- `F6` — старт / пауза
- `F7` — выход

## Какие модели использует проект

Основные файлы:
- `models/orange_click.onnx`
  основная кастомная модель детекции центров.
- `models/orange_click_refine.onnx`
  опциональная refine-модель для двухмодельной связки.
- `models/orange.png`
  шаблон апельсина.
- `models/minigame_start.png`
  шаблон старта мини-игры.

Если `orange_click_refine.onnx` отсутствует, проект всё равно может работать с одной основной ONNX-моделью плюс цвет и шаблоны.

## Как работает runtime

`OrangeBot` из `src/orange_bot/bot.py` на каждом цикле делает следующее:

1. Нажимает `E`.
2. Захватывает кадр.
3. Гонит кадр через:
   - кастомный ONNX-детектор;
   - цветовой detector;
   - шаблоны.
4. `MinigameStateTracker` решает, активна ли мини-игра.
5. `MainBrain` выбирает основные точки клика.
6. `Coverage Fill` добирает непрокрытые цветовые цели.
7. Бот кликает burst-серией.
8. После завершения делает шаг назад и возврат.

## Обучение новой модели

### 1. Построить click-датасет

```bat
scripts\build_click_dataset.bat
```

Или вручную:

```bat
python scripts/export_click_dataset.py --images-dir . --output-dir datasets/click_oranges
```

Скрипт:
- берёт скриншоты;
- прогоняет по ним текущий color detector;
- строит псевдоразметку точек клика;
- раскладывает данные по `train/val`.

### 2. Обучить модель

Быстрый вариант:

```bat
scripts\train_click_model.bat
```

Долгий вариант под VDS:

```bat
scripts\train_click_model_200.bat
```

Или вручную:

```bat
python scripts/train_click_model.py --data-dir datasets/click_oranges --epochs 200 --batch-size 2 --workers 0 --output-model models/orange_click.pt
```

### 3. Экспорт в ONNX

```bat
scripts\export_click_onnx.bat
```

Или вручную:

```bat
python scripts/export_click_onnx.py --checkpoint models/orange_click.pt --output models/orange_click.onnx
```

### 4. Проверить ONNX на скринах

```bat
scripts\infer_click_model.bat
```

Или вручную:

```bat
python scripts/infer_click_model.py --images-dir . --model models/orange_click.onnx --output-dir eval_click_model
```

Результат будет в папке `eval_click_model`:
- debug-изображения;
- `report.json`.

## Что лежит в click_model.py

`src/orange_bot/click_model.py` содержит:
- `OrangeClickNet`
  компактную encoder-decoder модель;
- `center_heatmap head`
  карту центров апельсинов;
- `orange_mask head`
  карту кликабельной области;
- postprocess для извлечения точек клика из heatmap и mask.

Это и есть новая основа вместо `YOLO`.

## Что настраивать

Все ключевые параметры лежат в:

```text
src/orange_bot/config.py
```

Там задаются:
- клавиши и тайминги;
- поведение кликов;
- параметры color detector;
- пороги brain;
- пути к ONNX-моделям;
- параметры legacy fallback.

## Текущее состояние проекта

Сейчас в репозитории уже есть:
- runtime-бот;
- интеграция кастомной ONNX-модели;
- код обучения новой модели;
- код экспорта в ONNX;
- код инференса по скринам.

Что ещё нужно для сильного результата:
- больше реальных скринов из игры;
- ручная чистка части лейблов;
- нормальное дообучение модели на VDS / GPU.

## Минимальный сценарий использования

Если нужен самый короткий путь:

1. Поставить зависимости:

```bat
install.bat
```

2. Обучить модель:

```bat
scripts\train_click_model_200.bat
```

3. Проверить ONNX:

```bat
scripts\infer_click_model.bat
```

4. Запустить бот:

```bat
start.bat
```

## Исходные скриншоты

Оригинальные скриншоты, которые использовались для сборки, проверки и обучения проекта, лежат отдельно в папке:

```text
samples/original_screenshots/
```

Это именно исходные кадры, а не debug-результаты и не eval-артефакты.
