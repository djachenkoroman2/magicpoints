# MagicPoints

## Назначение
`MagicPoints` — настольное приложение на Python (PyQt5 + OpenGL) для просмотра и анализа облаков точек.

Программа умеет:
- загружать облака точек из `TXT` и `PLY`;
- автоматически определять каналы `XYZ`, `label/class`, `RGB`;
- визуализировать большие наборы точек с использованием VBO;
- генерировать искусственные размеченные облака точек через встроенный генератор;
- сохранять сгенерированное облако в формате `PLY`.

## Ключевые функции
- Поддержка форматов:
  - `TXT` (пробелы/табуляция/запятые, с автоопределением колонок).
  - `PLY` (ASCII и binary).
- Режимы окраски:
  - по `label/class`;
  - по `RGB` (если доступно);
  - нейтральный цвет (если дополнительных каналов нет).
- Ограничение количества точек:
  - если файл больше `2,000,000` точек, выполняется равномерная случайная подвыборка.
- Навигация:
  - орбита/вращение (ЛКМ);
  - панорамирование (СКМ, либо `Shift` + ЛКМ);
  - zoom перетаскиванием (ПКМ) и колесом;
  - `Fit to View`, `Reset View`;
  - быстрые виды: `Top`, `Front`, `Left Side`, `Right Side`, `Back`, `Bottom`, `Front Isometric`.
- Генерация синтетического облака:
  - отдельное меню `Generate`;
  - отдельный диалог параметров;
  - автоматическая загрузка результата в окно;
  - сохранение сгенерированного облака в `PLY`.

## Структура интерфейса
- `File`:
  - `Open File`
  - `Exit`
- `Generate`:
  - `Generate Synthetic Cloud...`
  - `Save Generated Cloud as PLY...`
- `View`:
  - `Fit to View`
  - `Reset View`
  - `Top View`
  - `Front View`
  - `Left Side View`
  - `Right Side View`
  - `Back View`
  - `Bottom View`
  - `Front Isometric`
  - `Toggle RGB Mode`
- `Help`:
  - `About`

## Установка
Ниже показан рекомендованный вариант через `uv`.

### Windows
1. Установите `Python 3.10+`.
2. Установите `uv`:
```powershell
pip install uv
```
3. Перейдите в папку проекта:
```powershell
cd C:\path\to\visual
```
4. Создайте виртуальное окружение и установите зависимости:
```powershell
uv venv
uv pip install -r requirements.txt
```
5. Запустите приложение:
```powershell
uv run -- python main.py
```

### Linux (Ubuntu/Debian)
1. Установите системные библиотеки OpenGL/Qt:
```bash
sudo apt update
sudo apt install -y python3 python3-pip libgl1 libxkbcommon-x11-0 libxcb-xinerama0 libxcb-randr0 libxcb-shape0 libxcb-xfixes0 libxcb-render0
```
2. Установите `uv`:
```bash
pip install uv
```
3. Перейдите в папку проекта:
```bash
cd /path/to/visual
```
4. Создайте виртуальное окружение и установите зависимости:
```bash
uv venv
uv pip install -r requirements.txt
```
5. Запустите приложение:
```bash
uv run -- python main.py
```

## Запуск и использование
- Запуск:
```bash
uv run -- python main.py
```
- Запуск с автооткрытием файла:
```bash
uv run -- python main.py /path/to/cloud.ply
```

## Параметры командной строки
- `--max-points` — ограничение числа точек при загрузке файлов.

Пример:
```bash
uv run -- python main.py --max-points 1000000
```

## Зависимости
См. [requirements.txt](./requirements.txt):
- `numpy>=1.24`
- `PyOpenGL>=3.1`
- `PyQt5>=5.15`

## Примечания
- Для генерации используется файл `synthetic_labeled_point_cloud.py`.
- Генератор может печатать статистику в консоль (это нормальное поведение).
- Если на Linux не запускается Qt-плагин `xcb`, проверьте установку системных `libxcb*` пакетов из раздела установки.

## Копирайт
- Владелец: `Dyachenko Roman`
