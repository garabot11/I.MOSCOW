# Обнаружение препятствий в тоннеле метро по 3D-лидару (ROS 2 Humble, Docker)

Геометрический детектор: читает поток `sensor_msgs/PointCloud2` с лидара на голове поезда,
строит модель *нормального* тоннеля вокруг пути (полотно, стены, потолок, путевое
оборудование) и сообщает о любом компактном объекте внутри габарита поезда впереди вместе с
расстоянием до него. Без нейросети, без обучения, без GPU: NumPy + SciPy в одном Python-узле
ROS 2. То же ядро работает офлайн по bag-файлу для полной оценки.

English version of this file: `README.en.md`.

* Вход: любой rosbag2 (SQLite3 или MCAP) или живой топик с `PointCloud2` (обязательны поля
  x, y, z float32; в выданных записях 26-байтовая смешанная раскладка точки, см.
  `docs/data_notes.md`). Имя топика узнавать не нужно: по умолчанию (`input_topic:=auto`) узел
  подписывается на единственный топик `PointCloud2`, который появился в ROS-графе.
* Выход: `/obstacles/status` (`metro_obstacle_msgs/ObstacleStatus`: `UNKNOWN | CLEAR |
  DETECTED`, расстояние в метрах от лидара, данные по каждому объекту),
  `/obstacles/markers` (RViz), `/obstacles/candidates` (точки принятых кандидатов),
  необязательный JSON-lines-лог.
* Среда: Ubuntu 22.04 + ROS 2 Humble внутри образа (`ros:humble-ros-base-jammy`), только
  CPU, GPU не нужен; узел выдерживает ≈ 19 кадров/с для датчика 921 тыс. точек и ≈ 29 кадров/с
  для 307 тыс. на ноутбуке без привязки к ядрам, ≈ 12 / 19 кадров/с на его E-ядрах (раздел 6), 120–230 МБ ОЗУ
  на узел (замер); хосту нужен только Docker (и X11-сессия для RViz).

Документация: `docs/architecture.md`, `docs/algorithm.md`, `docs/experiments.md`,
`docs/data_notes.md` (на английском, с русской аннотацией в начале каждого файла). Разметка
для оценки: `annotations/events.yaml`; состав записей с SHA-256: `manifests/bags.yaml`.
Короткие видео работающей системы: `video/`.

## 1. Сборка

```bash
cd solution
docker build --target solution \
  --build-arg APP_UID="$(id -u)" --build-arg APP_GID="$(id -g)" \
  -t metro-lidar:final .
```

Сборка ставит все зависимости через apt (пакеты ROS, NumPy, SciPy, RViz2), копирует
исходники, выполняет `colcon build` и задаёт entrypoint, подключающий Humble и workspace.
`APP_UID/APP_GID` делают файлы в смонтированных папках результатов вашими. Сеть нужна только
при сборке. Сборка с нуля (`--no-cache --pull`) проверена 2026-09-29: ≈ 2,5 мин, итоговый образ
`sha256:69acee4e…` (см. `docs/experiments.md`, «Build record»). Проверка:

```bash
docker run --rm metro-lidar:final ros2 pkg list | grep metro_obstacle
docker run --rm metro-lidar:final python3 -m pytest -q /workspace/tests      # 38 тестов
```

Готовый образ без сборки и без сети: архив `metro-lidar-final.tar.gz` (462 МБ, SHA-256
`1c05211bbed1f4ecb62c6da302a56b7eff8c32a7c37cf57f42a39b428de1d313`, лежит в `../dist/` рядом с
папкой `solution/`):

```bash
sha256sum -c metro-lidar-final.tar.gz.sha256
docker load -i metro-lidar-final.tar.gz          # → metro-lidar:final, ID sha256:69acee4e…
```

В git-репозитории архив разрезан на части по 95 МБ; перед проверкой склейте их:
`cat metro-lidar-final.tar.gz.part-* > metro-lidar-final.tar.gz` (см. `../dist/README.md`).

## 2. Запуск (docker run → ros2 bag play → результат)

Примеры монтируют папку с bag-файлами в `/data` (только чтение) и папку результатов в
`/results`. «Bag» — это **директория** rosbag2 (`metadata.yaml` + `*.db3`). В `docker run -v`
путь хоста должен быть абсолютным (отсюда `$PWD/...`); скрипты из `scripts/` принимают и
относительные пути и сами приводят их к абсолютным.

### 2a. Без GUI: детектор + проигрывание, результат в терминале и в логе

Терминал 1 — детектор (печатает статус каждые 50 кадров; в логе одна JSON-строка на кадр):

```bash
mkdir -p results    # иначе Docker создаст папку от root, и узел не сможет писать лог
docker run --rm -it --name metro-lidar --network host --shm-size=1g \
  -e ROS_DOMAIN_ID=78 -e ROS_LOCALHOST_ONLY=1 \
  -v /path/to/bags:/data:ro -v "$PWD/results":/results \
  metro-lidar:final \
  ros2 launch metro_obstacle_detector detector.launch.py log_path:=/results/live.jsonl
```

Терминал 2 — проигрывание в том же контейнере (записанный QoS — *reliable*, с ним узел и
подписывается по умолчанию). Новый процесс `docker exec` не проходит через entrypoint
основного процесса, поэтому команда запускается через него явно — `/workspace/entrypoint.sh`
подключает Humble и workspace:

```bash
docker exec -it metro-lidar /workspace/entrypoint.sh \
  ros2 bag play /data/roundT_doubleT --rate 1.0 --delay 2 --read-ahead-queue-size 20 --wait-for-all-acked 5000
```

Терминал 3 — просмотр результата (одна строка на облако: статус, расстояние, время обработки):

```bash
docker exec -it metro-lidar /workspace/entrypoint.sh ros2 run metro_obstacle_detector status_monitor
docker exec -it metro-lidar /workspace/entrypoint.sh ros2 topic echo /obstacles/status   # полное сообщение
```

Проигрыватель Humble при старте без паузы сначала заполняет очередь чтения, а затем отправляет
первые 1–3 облака пачкой; подписка глубиной 1 оставляет из пачки последнее, поэтому эти кадры
не обрабатываются (особенность проигрывателя, не детектора; замерено 251 из 252 при
проигрывании этой командой). Полный учёт без этого эффекта — `scripts/run_stream_test.sh`
(раздел 5): там проигрыватель запускается с `--start-paused` и возобновляется сервисом
`/rosbag2_player/resume` после заполнения очереди.

Оболочки внутри образа тоже подключают окружение сами: `docker exec -it metro-lidar bash`
(интерактивная) и `docker exec metro-lidar bash -lc 'ros2 ...'` (login). Ещё до первого облака
узел раз в 0,5 с публикует `UNKNOWN` с диагностикой `no input: no cloud received on <топик>`
— так видно неверный топик/QoS или отсутствие лидара.

Топик облака узел находит сам: при `input_topic:=auto` (по умолчанию) он подписывается на
единственный топик `PointCloud2` в графе, как только проигрыватель его создаст (свой
`/obstacles/candidates` не учитывается; для издателя только с best-effort QoS подписка тоже
best-effort). `--delay 2` даёт узлу время подписаться до первого кадра. Если топиков облаков
несколько, узел не выбирает сам и сообщает их список в диагностике `UNKNOWN` — тогда укажите
`input_topic:=/sensing/lidar/hesai128/pointcloud` (или любой другой). Frame берётся из облака,
TF не нужен. Формат bag (SQLite3 / MCAP) определяется по `metadata.yaml`.

### 2b. С RViz2 (X11 на хосте)

`scripts/run_demo.sh` делает всё в одном контейнере — передаёт X11-cookie через отдельный
Xauthority-файл, берёт fixed frame из настоящего `header.frame_id` первого облака bag
(`bag_tools info --frame-id` в контейнере; переопределение — `FIXED_FRAME=<frame>`), запускает
узел с `rviz:=true fixed_frame:=<frame>` и проигрывает bag:

```bash
scripts/run_demo.sh /path/to/bags/doubleT_obstacle /sensing/lidar/hesai128/pointcloud 1.0 results/demo metro-lidar:final
scripts/run_demo.sh /path/to/bags/roundT_doubleT   /lidar_points                        1.0 results/demo metro-lidar:final
FIXED_FRAME=my_lidar scripts/run_demo.sh /path/to/bags/other_bag /points 1.0 results/demo   # явный frame
```

RViz показывает облако (цвет — высота), красным — точки принятых кандидатов, зелёную ось пути
с синими границами габарита, красный куб с подписью у каждого препятствия и текст статуса
(`CLEAR` / `OBSTACLE 55.7 m` / `UNKNOWN`). Используется программный OpenGL
(`LIBGL_ALWAYS_SOFTWARE=1`); с NVIDIA Container Toolkit добавьте `--gpus all` и
`-e NVIDIA_DRIVER_CAPABILITIES=graphics,display,utility`.

Ручной вариант: запустите контейнер как в 2a, добавив
`-e DISPLAY -e XAUTHORITY=/tmp/imos.xauth -e LIBGL_ALWAYS_SOFTWARE=1 -e QT_X11_NO_MITSHM=1
-v "$IMOS_XAUTH":/tmp/imos.xauth:ro -v /tmp/.X11-unix:/tmp/.X11-unix:ro`, и
`rviz:=true fixed_frame:=<header.frame_id облаков>` в launch (подготовка cookie — в
`scripts/run_demo.sh`; frame для bag печатает
`ros2 run metro_obstacle_detector bag_tools info --bag /data/<bag> --frame-id`).

### 2c. Офлайн-оценка всего bag (каждый кадр, без DDS)

```bash
docker run --rm -v /path/to/bags:/data:ro -v "$PWD/results":/results metro-lidar:final \
  ros2 run metro_obstacle_detector offline --bag /data/roundT_doubleT --out /results/roundT_doubleT.jsonl
```

Печатает строку на кадр и сводку (счётчики статусов, перцентили времени обработки, число
прочитанных сообщений против `metadata.yaml`, флаг `complete`); пишет `<name>.jsonl` и
`<name>_summary.json`; код возврата 3, если bag прочитан не до конца.
`scripts/eval_all_container.sh <bags> <results>` прогоняет все записи и проверяет каждую
(код контейнера, наличие файлов, `complete`, число строк JSONL): `ALLDONE` и код 0 только
при успехе всех, иначе `FAILED` со списком и код 1. `scripts/evaluate.py <results>`
сравнивает результаты с `annotations/events.yaml`.

### 2d. Проигрывание bag вне контейнера (на хосте или в другом контейнере)

Основной и проверенный путь — проигрывание в том же контейнере (2a). Если bag играется снаружи
(например, `ros2 bag play` установленного на стенде ROS 2), то по умолчанию обнаружение
топиков работает, а **данные не доходят**. Причина: Fast DDS для «соседей» на той же машине
выбирает разделяемую память, а у контейнера свой `/dev/shm`. Поэтому контейнер узла
запускается с профилем Fast DDS «только UDP» из образа и без `ROS_LOCALHOST_ONLY` (изоляция —
отдельным `ROS_DOMAIN_ID`, одинаковым у узла и проигрывателя):

```bash
docker run --rm -it --name metro-lidar --network host \
  -e ROS_DOMAIN_ID=78 -e FASTRTPS_DEFAULT_PROFILES_FILE=/workspace/config/fastdds_udp_only.xml \
  -v "$PWD/results":/results metro-lidar:final \
  ros2 launch metro_obstacle_detector detector.launch.py log_path:=/results/live.jsonl
# на хосте (ROS 2 Humble/Jazzy), без ROS_LOCALHOST_ONLY:
ROS_DOMAIN_ID=78 ros2 bag play /path/to/bags/roundT_doubleT --rate 1.0 --delay 2 --wait-for-all-acked 5000
```

Проверено: проигрыватель ROS 2 Jazzy на хосте — 252 из 252 кадров, второй Humble-контейнер —
251 из 252; без профиля — 0 кадров (`docs/architecture.md`, «Processes on the stand»).

## 3. Выход

`metro_obstacle_msgs/ObstacleStatus` (публикуется на каждое обработанное облако, а при
отсутствии входа дольше `stale_timeout_s` — и после запуска до первого облака, и после
прекращения потока — каждые 0,5 с со `STATUS_UNKNOWN`):

| поле | смысл |
|---|---|
| `header` | stamp = `header.stamp` входного облака (часы датчика), frame_id = frame облака |
| `status` | `0 UNKNOWN` (нет/некорректный/устаревший вход или полотно не наблюдается), `1 CLEAR`, `2 DETECTED` |
| `distance_m` | евклидово расстояние (м) от начала координат лидара до ближайшей достоверной поверхности ближайшего **подтверждённого** препятствия; иначе NaN |
| `forward_m` | его расстояние вдоль пути |
| `obstacles[]` | все кандидаты кадра: `confirmed`, `distance_m`, `forward_m`, `lateral_offset_m` (от оси пути), `height_above_rail_m`, размеры, `num_points`, `confidence`, `center` (в системе датчика) |
| `frame_index` | номер облака, **принятого** этим экземпляром узла (с 1); это не индекс кадра в bag — онлайн- и офлайн-результаты сопоставляются по `header.stamp` |
| `processing_ms` | время работы детектора над облаком (монотонные часы) |
| `rail_level_m`, `track_offset_m` | оценённый уровень рельса и смещение датчика от оси пути |
| `diagnostics` | причина UNKNOWN, сбросы, ошибки формата |

Расстояния отсчитываются от начала координат лидара (смещение до лобовой части поезда
неизвестно). JSON-лог (`log_path`) содержит также все отброшенные кластеры с причиной,
коэффициенты модели пути и время по стадиям.

## 4. Параметры и конфигурация

Все параметры детектора — в `config/detector.yaml` (комментарий к каждому ключу; файл
генерируется из `config.py` скриптом `scripts/gen_config.py`). Узел читает этот файл
(`config_file`), любой ключ можно переопределить ROS-параметром; офлайн-инструмент принимает
`--config` и `--set ключ=значение`.

```bash
# более широкий габарит и более раннее подтверждение
ros2 launch metro_obstacle_detector detector.launch.py input_topic:=/lidar_points \
   config_file:=/results/my_detector.yaml
ros2 run metro_obstacle_detector detector_node --ros-args -p input_topic:=/lidar_points \
   -p gauge_halfwidth_body:=1.5 -p confirm_hits:=1
ros2 run metro_obstacle_detector offline --bag /data/x --set gauge_halfwidth_body=1.5 --set confirm_hits=1
```

Аргументы launch: `input_topic` (`auto` по умолчанию или имя топика), `config_file`, `qos_reliability` (`reliable` по умолчанию —
записанный профиль; `best_effort` для датчиков, публикующих best-effort), `qos_depth`
(1 = самый свежий кадр), `log_path`, `use_sim_time`, `rviz`, `rviz_config`, `rviz_geometry`,
`fixed_frame` (fixed frame RViz = `header.frame_id` облаков; RViz-дисплей облака
переключается на `input_topic`).
Ключевые параметры (подробности в `docs/algorithm.md`):

| параметр | по умолчанию | ед. | эффект |
|---|---|---|---|
| `forward_axis` / `up_axis` | `-y` / `+z` | – | оси датчика: вдоль пути / вверх |
| `d_min` / `d_max` | 3 / 300 | м | анализируемая дальность |
| `gauge_halfwidth_low` / `gauge_halfwidth_body` | 1.0 / 1.40 | м | полуширина габарита ниже / выше 0,8 м над рельсом |
| `box_bottom_above_rail` / `box_top_above_rail` | 0.30 / 3.40 | м | вертикальные границы габаритной коробки |
| `min_points_near` / `min_points_far` | 5 / 4 | вокселей | минимальная поддержка кластера (< / ≥ 100 м) |
| `confirm_hits` / `confirm_window` | 2 / 3 | кадров | временное подтверждение (< 100 м; дальше `confirm_hits_far`/`confirm_window_far` = 3 / 4) |
| `stale_timeout_s` / `max_gap_s` | 1.5 / 1.5 | с | UNKNOWN при устаревшем входе / сброс состояния при разрывах |
| `report_beyond_support_m` | 15 | м | дальность выдачи за последней опорной выборкой модели пути |
| `debug_cloud` | true | – | публиковать `/obstacles/candidates` |
| `max_analysed_points` | 0 (выкл.) | точек | прореживание анализируемой области (каждая k-я точка, детерминированно) — запас по времени для плотных датчиков ценой опоры на дальних дистанциях, см. `docs/experiments.md` |
| `raw_decode` (параметр узла) | true | – | разбирать CDR облака без построения Python-сообщения (те же значения, меньше работы в потоке узла); `false` — стандартная десериализация rclpy |

QoS и время: подписчик — keep-last глубиной 1 (самый свежий кадр), *reliable* по умолчанию;
`use_sim_time` выключен (`header.stamp` облаков идёт по часам датчика с другой эпохой и
используется только для сопоставления, не для задержек). TF в выданных записях нет; все
выходы — в собственном frame облака.

## 5. Тесты и эксперименты

```bash
docker run --rm metro-lidar:final python3 -m pytest -q /workspace/tests
scripts/eval_all_container.sh /path/to/bags results/final_offline 1 metro-lidar:final
python3 scripts/evaluate.py results/final_offline --out results/final_offline/metrics.json
scripts/run_stream_test.sh /path/to/bags/roundT_doubleT results/stream /lidar_points
```

`run_stream_test.sh` проигрывает **весь** bag (`--rate 1.0`, QoS `reliable` — как у узла по
умолчанию; `QOS=best_effort`, `QOS_DEPTH=<n>` меняют профиль; необязательный 4-й аргумент —
предельная длительность, её срабатывание делает тест неуспешным) и пишет `coverage.json`:
отправленные кадры (= сообщения bag), обработанные (лог узла) и опубликованные (отдельный
подписчик `status_monitor --jsonl`), сопоставленные по `header.stamp`, с индексами
необработанных кадров и числом `UNKNOWN` до первого облака.

Результаты и протокол — в `docs/experiments.md`; сырые выходы в `results/` в образ не входят.

## 6. Известные ограничения и типовые проблемы

* Геометрия габарита (полуширина вагона 1,35 м, короба контактного рельса от 1,2 м, края
  платформ от 1,45 м) выведена из данных, а не задана организаторами; объекты ниже 0,3 м или
  между 1,0 и 1,35 м по ширине ниже 0,8 м высоты не сообщаются.
* Дальность ограничена тем, что тоннель позволяет увидеть: за поворотом коридор закрывается;
  на станциях и стрелках выдача прекращается там, где сечение тоннеля больше не опознаётся;
  низкие объекты — только там, где виден настил пути (≈ 60–120 м).
* `CLEAR` означает «в анализируемой области габарита нет подтверждённого объекта», а не
  доказанную свободу пути: область местами сужена (см. выше), а `d_max = 300 м` — граница
  обработки, не измеренная дальность обнаружения. Единственный пример препятствия стоит на
  ~56 м; обнаружение на 100–300 м на данных не подтверждено.
* Метрики (precision 0,979 / recall 1,0 по кадрам) получены на разработочных данных: пороги
  подбирались на всех шести записях, размечено одно событие с препятствием, 286 неоднозначных
  кадров исключены. Это не независимая оценка качества; организаторской разметки нет.
* `UNKNOWN` выдаётся, когда полотно не найдено (пустое/закрытое облако, неверные оси), и когда
  облаков нет — как после запуска до первого облака, так и после прекращения потока.
* Нет кадров? — диагностика `UNKNOWN` в `/obstacles/status` называет причину (нет топика,
  несколько топиков облаков, поток прекратился). Проверьте `ROS_DOMAIN_ID`,
  `ROS_LOCALHOST_ONLY`, имя топика (`ros2 topic list`) и QoS. Если bag играется вне контейнера
  узла — раздел 2d (профиль «только UDP», без `ROS_LOCALHOST_ONLY`).
* RViz «No transform»: задайте fixed frame равным `header.frame_id` облаков
  (`fixed_frame:=<frame>` в launch; frame bag печатает `bag_tools info --frame-id`);
  демо-скрипт читает его из bag сам. Жёлтый `Global Status: Warn` при этом ожидаем: TF в
  данных нет, облако и маркеры отображаются в собственном frame.
* Нет `ALLDONE` / код 1 у `eval_all_container.sh` — смотрите `<results>/FAILED` и
  `<results>/<bag>.log`.
* Время и ресурсы (замер в потоке 10 Гц, `docs/experiments.md`, «Throughput, CPU and slower
  cores»): обработка облака — медиана 24 мс (307 тыс. точек) и 48 мс (921 тыс.) на ноутбуке
  без привязки к ядрам, 39 / 73 мс на его E-ядрах; с декодированием сообщения узел тратит 34 / 54 мс CPU на
  кадр (≈ 29 / 19 кадров/с), на E-ядрах 53 / 84 мс (≈ 19 / 12 кадров/с). E-ядра взяты как замена
  более старых ядер стенда i7-9700E — это оценка, стенд не измерялся. Запас для плотного
  датчика на таких ядрах ≈ 1,2×; узел не следует ограничивать одним ядром (его потоки DDS
  конкурируют с детектором). Если кадр обрабатывается дольше интервала между кадрами, следующий
  заменяет ожидающий (политика «самый свежий кадр», глубина 1); такие кадры не обрабатываются,
  их число и индексы даёт `coverage.json` потокового теста. Посторонняя нагрузка на CPU
  уменьшает запас: до перехода на разбор CDR при загруженном браузером ноутбуке обрабатывался
  лишь каждый второй кадр плотного датчика (104 из 201). Прореживание (`max_analysed_points`)
  не помогает: дальний объект перестаёт обнаруживаться.
