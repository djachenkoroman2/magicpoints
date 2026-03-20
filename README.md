# MagicPoints

## Назначение
`MagicPoints` — настольное приложение на Python (`PyQt5` + `OpenGL`) для просмотра, анализа и базовой обработки облаков точек.

Проект умеет:
- загружать облака точек из `TXT` и `PLY`;
- автоматически определять каналы `XYZ`, `label/class`, `RGB`;
- визуализировать большие наборы точек через OpenGL VBO;
- генерировать искусственные размеченные облака точек;
- разделять размеченное облако на отдельные `PLY` по классам;
- выполнять кластеризацию `DBSCAN`;
- сохранять и загружать YAML-файлы с bounding boxes кластеров.

## Основные возможности

### Загрузка и отображение
- Поддерживаются форматы `TXT` и `PLY` (`ASCII` и `binary`).
- Если файл содержит поле `label/class`, доступна окраска по меткам.
- Если файл содержит `RGB`, доступна окраска по цвету.
- Если дополнительных каналов нет, используется нейтральная окраска.
- Для очень больших файлов действует ограничение загрузки: по умолчанию не более `2,000,000` точек с автоматической случайной подвыборкой.

### Навигация
- орбита/вращение: ЛКМ;
- панорамирование: СКМ или `Shift` + ЛКМ;
- приближение/отдаление: ПКМ drag и колесо мыши;
- для обычного режима просмотра вращение, панорамирование и `ПКМ drag` работают инверсно;
- в `Game Navigation Mode` управление мышью оставлено прежним;
- готовые виды: `Top`, `Front`, `Left Side`, `Right Side`, `Back`, `Bottom`, `Front Isometric`;
- `Fit to View`;
- `Reset View`;
- режим `Game Navigation Mode` с управлением `WASD` + мышь.

### Работа с размеченными облаками
- Команда `Edit -> Split...` делит текущее облако по значениям `label`.
- Для каждого класса сохраняется отдельный `PLY`-файл.
- Команда доступна только если в облаке есть поле метки.

### DBSCAN и кластеры
- Команда `Edit -> DBSCAN...` запускает кластеризацию текущего облака.
- Параметры `epsilon`, `MinPts` и путь к выходному YAML задаются в отдельном диалоге.
- Результат сохраняется в YAML-файл.
- Для каждого кластера сохраняются:
  - `id`;
  - `point_count`;
  - `bounding_box.min`;
  - `bounding_box.max`.
- После выполнения bounding boxes автоматически накладываются поверх текущего облака.
- Команда `File -> Open Clusters File` позволяет отдельно загрузить YAML-файл с кластерами и отобразить его как overlay.
- Команда `Edit -> Clear Viewport` полностью очищает viewport: удаляет текущее облако и все загруженные bounding boxes.

### Генерация синтетических облаков
- Команда `Generate -> Generate Exterior Synthetic Cloud...` открывает диалог с параметрами генерации.
- Поддерживается экспорт и импорт YAML-конфигурации генерации.
- После генерации результат автоматически загружается в главное окно.
- Команда `File -> Save Cloud as PLY...` сохраняет текущее облако точек, отображаемое во viewport, без overlay bounding boxes.

### Настройки проекта
- Команда `File -> Settings` открывает диалог настроек проекта.
- Настройки сохраняются в файл `settings.yaml` в корне проекта.
- В диалоге можно настроить:
  - каталог по умолчанию для сохранения файлов;
  - размер точки во viewport в диапазоне `1..10`;
  - цвет фона viewport через пресеты, `HEX` или `RGB`;
  - отображение bounding boxes: случайный или фиксированный цвет, толщину линии и показ `ID` возле бокса.

## Структура интерфейса
- `File`
  - `Open Point Cloud File`
  - `Open Clusters File`
  - `Save Cloud as PLY...`
  - `Save View as PNG...`
  - `Settings`
  - `Exit`
- `Edit`
  - `Split...`
  - `DBSCAN...`
  - `Clear Viewport`
- `Generate`
  - `Generate Exterior Synthetic Cloud...`
- `View`
  - `Fit to View`
  - `Reset View`
  - `Top View`
  - `Front View`
  - `Left Side View`
  - `Right Side View`
  - `Back View`
  - `Bottom View`
  - `Front Isometric`
  - `Game Navigation Mode`
  - `Toggle RGB Mode`
- `Help`
  - `About`

## Структура проекта
- [main.py](./main.py) — главное GUI-приложение.
- [utils/app_split_by_label.py](./utils/app_split_by_label.py) — CLI/API-утилита разделения размеченного облака по классам.
- [utils/synthetic_labeled_point_cloud.py](./utils/synthetic_labeled_point_cloud.py) — CLI/API-утилита генерации синтетического облака.
- [utils/app_dbscan_alg.py](./utils/app_dbscan_alg.py) — CLI/API-утилита кластеризации `DBSCAN`.
- [assets/icons](./assets/icons) — иконки приложения.
- [data](./data) — каталог по умолчанию для результатов и служебных файлов пользователя.
- [settings.yaml](./settings.yaml) — пользовательские настройки проекта.

## Каталог `data`
По умолчанию новые результаты сохраняются в каталог `data/`, если в `settings.yaml` не указан другой путь:
- результаты `DBSCAN`;
- YAML-конфигурации генерации;
- сохраненные синтетические облака;
- стандартные результаты CLI-генератора;
- другие рабочие файлы, создаваемые из GUI.

Каталог `data/` добавлен в `.gitignore` и не попадает в репозиторий.

## Установка
Ниже показан рекомендованный вариант через `uv`.

### Windows
1. Установите `Python 3.10+`.
2. Установите `uv`:

```powershell
pip install uv
```

3. Перейдите в каталог проекта:

```powershell
cd C:\path\to\magicpoints
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

3. Перейдите в каталог проекта:

```bash
cd /path/to/magicpoints
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

## Запуск приложения

Обычный запуск:

```bash
uv run -- python main.py
```

Запуск с автооткрытием файла:

```bash
uv run -- python main.py /path/to/cloud.ply
```

Запуск с ограничением числа загружаемых точек:

```bash
uv run -- python main.py --max-points 1000000
```

## CLI-утилиты

### Split по меткам

Справка:

```bash
uv run -- python utils/app_split_by_label.py --help
```

Пример запуска:

```bash
uv run -- python utils/app_split_by_label.py /path/to/labeled_cloud.ply --prefix split --output-dir data
```

### Генератор синтетического облака

Справка:

```bash
uv run -- python utils/synthetic_labeled_point_cloud.py --help
```

Пример запуска:

```bash
uv run -- python utils/synthetic_labeled_point_cloud.py --total-points 100000 --seed 12 --no-visualize
```

По умолчанию CLI-генератор сохраняет результаты в `data/`.

### DBSCAN

Справка:

```bash
uv run -- python utils/app_dbscan_alg.py --help
```

Пример запуска:

```bash
uv run -- python utils/app_dbscan_alg.py /path/to/cloud.ply --epsilon 1.0 --min-pts 8 --output data/cloud_dbscan.yaml
```

## Использование API

Обе утилиты можно подключать из Python-кода:

```python
from utils import app_split_by_label as split_module
from utils import synthetic_labeled_point_cloud as synthetic_module
from utils import app_dbscan_alg as dbscan_module
```

Пример вызова `Split` через API:

```python
result = split_module.split_point_cloud_by_label_arrays(
    points=points_xyz,
    labels=labels,
    prefix="split",
    output_dir="data",
)
```

Пример вызова `DBSCAN` через API:

```python
result = dbscan_module.run_dbscan_on_points(points_xyz, epsilon=1.0, min_pts=8)
```

Пример генерации облака:

```python
cloud = synthetic_module.generate_point_cloud(total_points=100000, seed=12)
```

## Форматы данных

### Облако точек
- вход GUI: `TXT`, `PLY`;
- вход DBSCAN CLI: `TXT`, `PLY`.

### Кластеры
- формат: `YAML`;
- используется для сохранения и загрузки bounding boxes кластеров.

Пример структуры:

```yaml
schema: magicpoints.dbscan.clusters.v1
epsilon: 1.0
min_pts: 8
source_point_count: 100000
clusters:
  - id: 0
    point_count: 125
    bounding_box:
      min: [0.0, 1.0, 2.0]
      max: [4.0, 5.0, 6.0]
```

## Зависимости
См. [requirements.txt](./requirements.txt):
- `numpy>=1.24`
- `PyOpenGL>=3.1`
- `PyQt5>=5.15`
- `PyYAML>=6.0`

## Примечания
- Утилиты командной строки находятся в каталоге `utils/`.
- Генератор и DBSCAN доступны как из CLI, так и через import/API.
- Генератор может печатать статистику в консоль — это нормальное поведение.
- Если на Linux не запускается Qt-плагин `xcb`, проверьте установку системных `libxcb*` пакетов из раздела установки.

## Копирайт
- Владелец: `Dyachenko Roman`
