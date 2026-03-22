# MagicPoints

`MagicPoints` — настольное приложение на Python (`PyQt5` + `OpenGL`) для просмотра, анализа и базовой обработки облаков точек и треугольных мешей. Проект поддерживает загрузку `TXT` и `PLY`, автоматическое определение каналов `XYZ` / `label` / `RGB`, открытие mesh-файлов `PLY`, визуализацию больших наборов точек, `DBSCAN`-кластеризацию и генерацию синтетических размеченных сцен.

## Возможности

- загрузка облаков точек из `TXT` и `PLY` (`ASCII` и `binary`);
- открытие треугольных mesh-файлов из `PLY` через `File -> Open Mesh File` с проверкой, что файл содержит именно mesh;
- автоопределение координат, меток классов и `RGB`-каналов;
- отрисовка больших облаков через OpenGL VBO;
- автоматическая случайная подвыборка при загрузке очень больших файлов;
- разбиение размеченного облака на отдельные `PLY`-файлы по значениям `label`;
- `DBSCAN`-кластеризация с сохранением кластеров в `YAML`;
- загрузка и отображение bounding boxes поверх текущего облака;
- генерация синтетического размеченного облака точек из GUI и CLI;
- сохранение текущего облака, текущей mesh-поверхности в `PLY` и текущего вида в `PNG`;
- проектные настройки: каталог вывода, размер точки, фон viewport, стиль bounding boxes.

## Быстрый старт

### Требования

- `Python 3.10+`;
- видеодрайвер с поддержкой `OpenGL 2.1+`;
- зависимости из [requirements.txt](./requirements.txt).

### Установка через `uv`

#### Windows

```powershell
py -m pip install uv
cd C:\path\to\magicpoints
uv venv
uv pip install -r requirements.txt
uv run -- python main.py
```

#### Linux (Ubuntu/Debian)

Перед установкой Python-зависимостей желательно поставить системные библиотеки Qt/OpenGL:

```bash
sudo apt update
sudo apt install -y python3 python3-pip libgl1 libxkbcommon-x11-0 libxcb-xinerama0 libxcb-randr0 libxcb-shape0 libxcb-xfixes0 libxcb-render0
python3 -m pip install uv
cd /path/to/magicpoints
uv venv
uv pip install -r requirements.txt
uv run -- python main.py
```

### Альтернатива через стандартный `venv`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Для Windows в активированном окружении используйте:

```powershell
.venv\Scripts\activate
python main.py
```

## Запуск GUI

Обычный запуск:

```bash
uv run -- python main.py
```

Открыть файл сразу после старта:

```bash
uv run -- python main.py /path/to/cloud.ply
```

Ограничить число загружаемых точек:

```bash
uv run -- python main.py --max-points 1000000
```

По умолчанию приложение загружает не более `2_000_000` точек. Если файл больше, используется случайная подвыборка.

## Основные сценарии работы

### Просмотр и навигация

- вращение: `ЛКМ`;
- панорамирование: `СКМ` или `Shift + ЛКМ`;
- приближение и отдаление: колесо мыши или `ПКМ drag`;
- предустановленные виды: `Top`, `Front`, `Left Side`, `Right Side`, `Back`, `Bottom`, `Front Isometric`;
- команды `Fit to View` и `Reset View`;
- режим `Game Navigation Mode` с управлением `WASD` + мышь.

### Работа с размеченными облаками

Если во входном облаке есть поле `label` / `class`, команда `Tools -> Split...` сохранит отдельный `PLY` для каждого класса. Если есть `RGB`, можно переключать режим отображения цвета через `View -> Toggle RGB Mode`.

### DBSCAN и overlay кластеров

Команда `Tools -> DBSCAN...` запускает кластеризацию текущего облака, сохраняет результат в `YAML` и автоматически отображает bounding boxes найденных кластеров поверх сцены. Отдельный `YAML` с кластерами можно загрузить через `File -> Open Clusters File`.

Каждый кластер хранит:

- `id`;
- `point_count`;
- `bounding_box.min`;
- `bounding_box.max`.

### Работа с mesh

Команда `File -> Open Mesh File` открывает только `PLY`, в которых есть вершины и face-элементы треугольного меша. `PLY` без face-данных, облака точек и файлы bounding boxes этой командой не загружаются. Для уже открытого меша сразу применяются настройки из `File -> Settings -> Mesh Display`.

### Генерация синтетического облака

Команда `Generate -> Generate Exterior Synthetic Cloud...` открывает диалог генерации синтетической сцены. Конфигурацию генерации можно импортировать и экспортировать в `YAML`, а результат после генерации сразу загружается в viewport.

### Настройки проекта

Команда `File -> Settings` управляет параметрами проекта:

- каталогом по умолчанию для результатов;
- размером точки (`1..10`);
- цветом фона viewport;
- цветом, толщиной линий и показом `ID` для bounding boxes;
- вкладкой `Mesh Display` с режимом отрисовки, colormap и сглаживанием нормалей для сетки.

## CLI-утилиты

Все утилиты находятся в каталоге [utils](./utils).

### Split по меткам

Справка:

```bash
uv run -- python utils/app_split_by_label.py --help
```

Пример:

```bash
uv run -- python utils/app_split_by_label.py /path/to/labeled_cloud.ply --prefix split --output-dir data
```

Утилита принимает `TXT` или `PLY`, ищет поле метки и сохраняет по одному `PLY`-файлу на каждый класс.

### DBSCAN

Справка:

```bash
uv run -- python utils/app_dbscan_alg.py --help
```

Пример:

```bash
uv run -- python utils/app_dbscan_alg.py /path/to/cloud.ply --epsilon 1.0 --min-pts 8 --output data/cloud_dbscan.yaml
```

Результат сохраняется в `YAML`. Если расширение не указано, утилита автоматически добавит `.yaml`.

### Генератор синтетического облака

Справка:

```bash
uv run -- python utils/synthetic_labeled_point_cloud.py --help
```

Минимальный пример:

```bash
uv run -- python utils/synthetic_labeled_point_cloud.py --total-points 100000 --seed 12 --no-visualize
```

Пример с конфигурацией:

```bash
uv run -- python utils/synthetic_labeled_point_cloud.py --config /path/to/generation.yaml --no-visualize
```

CLI-генератор поддерживает большой набор параметров для рельефа, состава классов, количества объектов, форм крон, параметров зданий, кустарников и травяных пятен.

## Использование как API

Утилиты можно использовать как обычные Python-модули:

```python
from utils import app_dbscan_alg as dbscan_module
from utils import app_split_by_label as split_module
from utils import synthetic_labeled_point_cloud as synthetic_module
```

Пример `Split`:

```python
result = split_module.split_point_cloud_by_label_arrays(
    points=points_xyz,
    labels=labels,
    prefix="split",
    output_dir="data",
)
```

Пример `DBSCAN`:

```python
result = dbscan_module.run_dbscan_on_points(
    points_xyz,
    epsilon=1.0,
    min_pts=8,
    output_path="data/cloud_dbscan.yaml",
)
```

Пример генерации:

```python
cloud = synthetic_module.generate_point_cloud(total_points=100000, seed=12)
```

`generate_point_cloud(...)` возвращает массив формы `(N, 4)`, где столбцы соответствуют `x, y, z, label`.

## Форматы данных

### Входные данные

- GUI: `TXT`, `PLY`;
- `app_split_by_label.py`: `TXT`, `PLY`;
- `app_dbscan_alg.py`: `TXT`, `PLY`;
- генератор: параметры CLI или `YAML`-конфигурация.

### Выходные данные

- сохранение облака из GUI: `PLY`;
- сохранение вида из GUI: `PNG`;
- split по меткам: набор `PLY`;
- `DBSCAN`: `YAML` с описанием кластеров;
- конфигурация генератора: `YAML`.

Пример структуры файла кластеров:

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

## `settings.yaml`

Файл [settings.yaml](./settings.yaml) хранит пользовательские настройки проекта. Если путь `output_directory` относительный, он интерпретируется относительно корня репозитория.

Поля файла:

- `output_directory`;
- `point_size`;
- `viewport_background`;
- `bounding_box_color_mode`;
- `bounding_box_color`;
- `bounding_box_line_width`;
- `bounding_box_show_id`.

Если `settings.yaml` отсутствует или поврежден, приложение использует встроенные значения по умолчанию.

## Структура проекта

- [main.py](./main.py) — GUI-приложение и основная логика визуализации.
- [utils/app_split_by_label.py](./utils/app_split_by_label.py) — split размеченного облака на отдельные `PLY`.
- [utils/app_dbscan_alg.py](./utils/app_dbscan_alg.py) — `DBSCAN` для `TXT` и `PLY`.
- [utils/synthetic_labeled_point_cloud.py](./utils/synthetic_labeled_point_cloud.py) — генератор синтетических размеченных сцен.
- [assets/icons](./assets/icons) — иконки интерфейса.
- [requirements.txt](./requirements.txt) — Python-зависимости проекта.
- [settings.yaml](./settings.yaml) — пользовательские настройки.

## Каталог `data`

Каталог `data/` не хранится в репозитории и создается по мере необходимости. По умолчанию туда попадают:

- результаты `DBSCAN`;
- экспортированные `PLY` после split;
- конфигурации генератора;
- синтетические облака, созданные через GUI или CLI;
- другие рабочие файлы, если в `settings.yaml` не указан другой путь.

`data/` добавлен в `.gitignore`.

## Примечания

- Для `TXT`-файлов поддерживается автоопределение колонок по заголовку, если он есть.
- При отсутствии `label` split по классам недоступен.
- При отсутствии `RGB` переключение в цветовой режим по каналам не имеет эффекта.
- Если на Linux появляется ошибка Qt-плагина `xcb`, проверьте установку `libxcb*` пакетов из раздела установки.

## Копирайт

Владелец проекта: `Dyachenko Roman`
