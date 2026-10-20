# Обнаружение препятствий в тоннеле метро по 3D-лидару (ROS 2 Humble, Docker)

Узел ROS 2 читает поток `sensor_msgs/PointCloud2` с лидара на голове поезда и строит модель
*нормального* тоннеля вокруг пути: полотно, стены, потолок, путевое оборудование. Внутри габарита
поезда работают три детектора:

* Д1 — кластеры точек в габарите (NumPy);
* Д2 — лучевой детектор на range-image;
* Д3 — нейросеть U-Net на range-image (PyTorch, `models/d3_unet.pt`).

Их голоса объединяет взвешенное слияние с подтверждением во времени. Узел сообщает о препятствии и
расстоянии до него, а также о том, до какой дальности путь проверен.

**Основной режим — GPU** (CUDA через PyTorch). **CPU — только аварийный режим.** В нём решение
громко предупреждает организаторов: что с GPU нейросетевой детектор Д3 работает в каждом кадре и частота
выше, почему GPU недоступен, как
это исправить и как связаться с командой IRS-MIREA (Telegram: IRS-MIREA). То же ядро работает
офлайн по bag-файлу для полной оценки.

English version of this file: `README.en.md`; English summary: the end of this file and of `docs/launch.md`.

* **Вход:** любой rosbag2 (SQLite3 или MCAP) или живой топик `PointCloud2`. Поддерживаются Hesai
  307 тыс. и 921 тыс. точек, Seyond Falcon Prime и синтетические сцены. Имя топика узнавать не нужно:
  `input_topic:=auto` по умолчанию.
* **Выход:**
  * `/obstacles/status` (`metro_obstacle_msgs/ObstacleStatus`): статус, расстояние, «путь проверен
    до X м», режим GPU/CPU и данные по каждому объекту;
  * `/obstacles/markers` для RViz;
  * `/obstacles/candidates`;
  * JSON-lines-лог.
* **Среда.** Всё нужное находится в одном образе (раздел 1):
  * Ubuntu 22.04 + ROS 2 Humble, rosbag2, RViz2;
  * PyTorch 2.11.0+cu128 со своими библиотеками CUDA — с хоста нужен только драйвер NVIDIA ≥ 570;
  * Mesa для программной отрисовки;
  * команда `metro`.

  Хосту нужен Docker, для GPU — NVIDIA Container Toolkit, для RViz — X11-сессия.

Документация:

* запуск — `docs/launch.md` (пошагово: подготовка хоста, проверка GPU, сценарий показа, ошибки и их исправление);
* `docs/criteria_ru.md` — критерии экспертов: что сделано и чем подтверждено;
* `docs/architecture.md`, `docs/algorithm.md`, `docs/experiments.md`, `docs/data_notes.md`;
* модель Д3 — `models/model_card.md`;
* разметка — `annotations/events.yaml`.

## 1. Получение образа

**Основной способ — готовый архив, без сборки.** Архив образа (4,5 ГБ) лежит в
[релизе v3](https://github.com/garabot11/I.MOSCOW/releases/tag/v3) частями по 95 МБ; контрольные суммы — в
`../dist/`. Скачивание и загрузка — `../dist/README.md` и `docs/launch.md` §3:

```bash
cd dist   # 45 частей metro-lidar-final.tar.gz.part-NN из релиза v3
sha256sum -c metro-lidar-final.parts.sha256                      # проверка частей
cat metro-lidar-final.tar.gz.part-* | gunzip | docker load       # → metro-lidar:gpu-dev, ID a0e1e7e33476
```

Архив создаётся скриптом `../dist/make_archive.sh` (`docker save | gzip`, SHA-256, части).

**Альтернатива — сборка из `Dockerfile`.** Нужен интернет (около 4,5 ГБ загрузки PyTorch/CUDA).
GPU и BuildKit не нужны.

```bash
cd solution
docker build -t metro-lidar:gpu-dev .
docker run --rm metro-lidar:gpu-dev python3 -m pytest -q /workspace/tests     # проверка
```

Образ на диске занимает около 13 ГБ. Версии всех Python-пакетов зафиксированы в
`docker/requirements-torch.txt`.

## 2. Запуск

Всё делается одной командой на хосте, из папки `solution/`:

```bash
./run.sh --data /путь/к/датасету          # можно повторить: --data A --data B → /data/A, /data/B
```

`run.sh` выполняет следующее:

* сам определяет, можно ли дать контейнеру GPU (`nvidia-smi`, runtime `nvidia`, пробный
  `docker run --gpus all`), и добавляет `--gpus all -e NVIDIA_DRIVER_CAPABILITIES=all`;
* запускает постоянный контейнер `metro-lidar`. Данные монтируются только для чтения в `/data`,
  результаты — в `/results` (по умолчанию `~/metro-lidar-results`);
* выполняет самопроверку `doctor`;
* запускает детектор и RViz: на GPU, через PRIME на гибридном ноутбуке или с программной
  отрисовкой. Без дисплея RViz не запускается;
* показывает живую консоль статуса. Сочетание Ctrl-C закрывает только консоль.

Второй терминал:

```bash
./run.sh bags                                    # все записи под /data (с подпапками; синтетика — как <набор>/<сцена>)
./run.sh play doubleT_obstacle                   # по имени папки, пути относительно /data или полному пути
./run.sh play for_hackathon/roundT_doubleT 0.5   # скорость 0.5
./run.sh play val_hesai_insert/val_hesai_insert_001_pos   # синтетическая сцена (<сцена>/bag)
./run.sh status | console | doctor | stop
```

* **Клавиши при проигрывании:** ПРОБЕЛ — пауза и продолжение, ↑/↓ — скорость ±10 %, → — следующее
  сообщение.
* **Смена записи без перезапуска.** Просто выполните `./run.sh play <другая запись>`: предыдущая
  остановится, детектор сбросит состояние на скачке времени.
* **Топик облака.** Проигрывается только топик облака, он публикуется в `/lidar_points`.
* **Фрейм.** Тождественный TF сводит `frame_id` записи (`lidar_livox`, `seyond`) к fixed frame RViz
  `hesai_lidar`.

Ключи `run.sh`:

| Ключ | Что делает |
|---|---|
| `--cpu` | Принудительный режим CPU |
| `--no-gui` | Без RViz |
| `--udp` | Bag проигрывается вне контейнера |
| `--cpuset-cpus 0-7` | Восемь ядер, как на стенде |
| `--results DIR` | Папка результатов |
| `--image IMG` | Другой образ |

### Режимы GPU и CPU, баннер

| Режим | Когда | Что видно |
|---|---|---|
| `gpu` | GPU доступен и прошёл пробный расчёт | `mode` = `gpu`, Д3 на GPU (Д1 и Д2 — всегда CPU) |
| `cpu_emergency` | GPU нет или он сломался во время работы | См. ниже |
| `cpu_forced` | `./run.sh --cpu` | Баннер один раз |

В режиме `cpu_emergency` Д3 горячо переключается на CPU, и предупреждение дублируется везде:

* баннер на русском и английском при старте и каждые 5 минут в консоли;
* красная надпись в RViz;
* поля `mode` и `mode_reason` в статусе;
* файл `/results/ВНИМАНИЕ_GPU.txt`.

Баннер говорит, что GPU недоступен и решение работает в аварийном режиме на CPU; что с GPU
нейросетевой детектор Д3 работает в каждом кадре (дополнительные свидетельства, на Falcon +22 кадра
дальнего прохода) и частота выше; на CPU Д3 считается каждый третий кадр. В нём есть причина, точные команды исправления
(установка NVIDIA Container Toolkit, запуск с `--gpus all`, драйвер ≥ 570) и контакт: «Свяжитесь с
командой IRS-MIREA в Telegram: IRS-MIREA».

Сборка PyTorch содержит ядра sm_75…sm_120. RTX 4070 Ti SUPER стенда (sm_89) выполняет ядра sm_86
по двоичной совместимости CUDA. `doctor` печатает список архитектур и capability, пробный расчёт
при старте подтверждает работу.

### Ручной запуск (без `run.sh`)

```bash
mkdir -p results      # иначе Docker создаст её от root, и логи не запишутся
docker run -d --name metro-lidar --init --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all \
  --network host --shm-size=1g -v /path/to/bags:/data:ro -v "$PWD/results":/results \
  metro-lidar:gpu-dev sleep infinity
docker exec -it metro-lidar metro start --no-rviz              # самопроверка, детектор, консоль
docker exec -it metro-lidar metro play roundT_doubleT          # второй терминал
docker rm -f metro-lidar                                       # остановить
```

Без `--gpus all` тот же запуск работает в аварийном режиме CPU с баннером.

### Офлайн-оценка всего bag (каждый кадр, без DDS)

```bash
docker run --rm --gpus all -v /path/to/bags:/data:ro -v "$PWD/results":/results metro-lidar:gpu-dev \
  ros2 run metro_obstacle_detector offline --bag /data/roundT_doubleT --out /results/roundT_doubleT.jsonl
```

Команда печатает строку на кадр и сводку и пишет `<name>.jsonl` и `<name>_summary.json`. Ключи
`--device cpu` и `--no-d3` выбирают устройство и отключают нейросеть. Чтобы прогнать все записи,
используйте `scripts/eval_all_container.sh <bags> <results>`, сравнить с разметкой —
`scripts/evaluate.py <results>`.

### Проигрывание bag вне контейнера

Fast DDS между контейнером и хостом выбирает разделяемую память и молча не доставляет данные.
Поэтому:

* `./run.sh --udp` включает профиль «только UDP»;
* `ROS_DOMAIN_ID` берётся с хоста (по умолчанию 0);
* `ROS_LOCALHOST_ONLY` не устанавливается.

```bash
ros2 bag play /path/to/bags/roundT_doubleT          # на хосте, тот же ROS_DOMAIN_ID
```

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
| `verified_to_m` | до какой дальности в этом кадре построен и проверен коридор габарита («путь проверен до X м»; NaN, если нет); `CLEAR` относится только к этому участку |
| `static_fraction` | доля пикселей range-image без изменений с прошлого кадра (1 = поезд стоит) |
| `mode` | режим вычислений детекторов: `gpu`, `cpu_emergency`, `cpu_forced`, `cpu` |
| `mode_reason` | почему не GPU (пусто в режиме `gpu`) |

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

Аргументы launch: `input_topic` (`auto` по умолчанию или имя топика), `config_file`,
`device` (`auto` | `gpu` | `cpu`), `team_contact` (контакт в баннере), `results_dir` (папка для
`ВНИМАНИЕ_GPU.txt`; при падении узел перезапускается сам — `respawn`), `qos_reliability` (`reliable` по умолчанию —
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
| `confirm_hits` / `confirm_window` | 2 / 3 | кадров | **не действует при включённом слиянии (`fusion_enabled`, по умолчанию)** — решают треки слияния; временное подтверждение (< 100 м; дальше `confirm_hits_far`/`confirm_window_far` = 3 / 4) |
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
docker run --rm metro-lidar:gpu-dev python3 -m pytest -q /workspace/tests
solution/docker/failure_matrix.sh --data /path/to/dataset          # матрица отказов запуска (run.sh + metro)
scripts/eval_all_container.sh /path/to/bags results/final_offline 1 metro-lidar:gpu-dev
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
* Дальность ограничена тем, что тоннель позволяет увидеть: за поворотом коридор закрывается
  (а там, где стена «входит» в прямолинейно продолженный коридор, — обрезается, см.
  `docs/algorithm.md` §3); на станциях и стрелках выдача прекращается там, где сечение тоннеля
  больше не опознаётся; низкие объекты — только там, где виден настил пути (≈ 60–120 м), выше
  настила уровень рельса продолжается по потолку (до 150–200 м в прямых тоннелях).
  Поле `verified_to_m` в сообщении и подпись «track checked to N m» в RViz показывают, до какой
  дальности коридор в этом кадре построен и проверен; `CLEAR` означает «до этой дальности
  подтверждённого объекта нет», а не доказанную свободу пути до `d_max = 300 м`.
* Подтверждение требует, чтобы **Д1 и Д2 проголосовали оба** (Д3 нейросеть добавляет вес, но не
  голосует; подтверждённый трек затем удерживается и слабыми попаданиями Д1/Д2 в последних 5 кадрах)
  на любой дальности; объект, который видит
  только один детектор (например, тёмный человек, которого Д2 замечает на 2 кадра раньше Д1),
  сообщается только после появления второго. Это цена нуля ложных тревог на шести записях.
* Проверенные дальности: единственное реальное препятствие (56 м) — с кадра 11; тёмный человек
  на записи Falcon Prime впервые подтверждён на **154 м** (кадр 2727, одинаково с Д3 и без), стоя на 100 м
  виден в 510 из 510 кадров, при подходе со 100 до 4 м — в 710 из 710 (дальний проход 150 → 100 м: 390 из 451
  кадра); в `cloud_with_fake_obj` (стиль вставки объектов, которым организаторы будут тестировать)
  подтверждаются стена с 76 м, низкая плита с 91 м и кубы 0,3 м с 37–52 м, а боковые ящики, ящики над путём и
  висящий столб (заходят в габарит краем) — нет. На синтетике организаторов с разметкой найдены все 7 объектов
  внутри нашего габарита (49 % объект-кадров) и 8 из 13 по всему их списку (35 %), 0 ложных; 6 объектов стоят
  в 1,1–2,3 м от оси ниже 0,85 м — вне нашего габарита. На нашей синтетике (`docs/experiments.md`) полнота на
  100–150 м 61 % для вставленных в реальные кадры людей и коробок и ниже для параметрических сцен; объекты
  дальше ≈ 150 м (Hesai) и тёмные цели дальше ≈ 150 м (Falcon) дают слишком мало отражений.
* `UNKNOWN` выдаётся, когда полотно не найдено (пустое/закрытое облако, неверные оси), и когда
  облаков нет — как после запуска до первого облака, так и после прекращения потока.
* Нет кадров? — диагностика `UNKNOWN` в `/obstacles/status` называет причину (нет топика,
  несколько топиков облаков, поток прекратился). Проверьте `ROS_DOMAIN_ID`,
  `ROS_LOCALHOST_ONLY`, имя топика (`ros2 topic list`) и QoS. Если bag играется вне контейнера
  узла — `docs/launch.md` §12.3 (профиль «только UDP», без `ROS_LOCALHOST_ONLY`).
* RViz «No transform»: задайте fixed frame равным `header.frame_id` облаков
  (`fixed_frame:=<frame>` в launch; frame bag печатает `bag_tools info --frame-id`);
  демо-скрипт читает его из bag сам. Жёлтый `Global Status: Warn` при этом ожидаем: TF в
  данных нет, облако и маркеры отображаются в собственном frame.
* Нет `ALLDONE` / код 1 у `eval_all_container.sh` — смотрите `<results>/FAILED` и
  `<results>/<bag>.log`.
* Время и ресурсы (`docs/experiments.md`, «Performance»; ноутбук, стенд не измерялся): на 8 закреплённых
  ядрах три детектора вместе — медиана / p95 49–53 / 65–70 мс на кадр для **движущегося** поезда с облаком
  307 тыс. точек (формат стенда; запись с платформой и стрелкой), 27 / 39 мс для стоящего поезда с 921 тыс.
  точек (модель заморожена), 53 / 56 мс для Falcon Prime, 64–66 / 71–75 мс для отредактированных
  неорганизованных облаков организаторов; **ни одного кадра дольше 100 мс из 3480**. Д3 на GPU стоит 8–21 мс и
  считается параллельно с Д2, в аварийном режиме CPU — каждый третий кадр. Ядра стенда (i7-9700E, 2,6 ГГц)
  медленнее — ожидаем ≈ 1,3–1,5× от этих чисел: на отредактированных облаках ≈ 95–110 мс p95, то есть часть
  кадров может пропускаться. Если кадр обрабатывается дольше интервала между кадрами, следующий заменяет
  ожидающий (политика «самый свежий кадр», глубина 1). Узел не следует ограничивать одним ядром (его потоки
  DDS конкурируют с детектором); прореживание (`max_analysed_points`) не помогает — дальний объект перестаёт
  обнаруживаться.

## English summary

* **What it does:** obstacle detection ahead of a metro train from a forward 3D LiDAR (ROS 2 Humble).
  Three detectors are fused: D1 point clusters, D2 a beam detector on the range image and D3 a
  range-image U-Net (`models/d3_unet.pt`, PyTorch). The status reports the obstacle distance, the
  range up to which the track was checked (`verified_to_m`) and the compute mode (`mode`,
  `mode_reason`).
* **Get the image:**
  * primary: the 45 parts (95 MB) of the archive in release
    [v3](https://github.com/garabot11/I.MOSCOW/releases/tag/v3), checksums in `../dist`
    (`sha256sum -c metro-lidar-final.parts.sha256`, then `cat metro-lidar-final.tar.gz.part-* | gunzip | docker load`);
  * alternative: `docker build -t metro-lidar:gpu-dev solution` (needs internet, no GPU, no
    BuildKit).
* **Run:**
  * `./run.sh --data <dataset>` starts the persistent container. It uses the GPU automatically when
    the host allows it, runs the self-check, starts the detector and RViz, and shows the console.
  * `./run.sh play <bag|set/scene> [rate]`: SPACE pauses, UP/DOWN changes the rate. Another
    `play` switches bags without a restart.
  * Other commands: `./run.sh bags | status | doctor | stop`.
* **GPU first, CPU only in an emergency:** without a usable GPU the solution runs on the CPU and
  warns the organisers. The RU+EN banner appears at start and every 5 minutes, there is a red text
  in RViz, the `mode` field and `/results/ВНИМАНИЕ_GPU.txt`. The banner gives the reason, the exact
  fix and the contact "Contact team IRS-MIREA on Telegram: IRS-MIREA".
* **More:** step-by-step launch, GPU check, demo scenario and troubleshooting are in `docs/launch.md`.
