# Запуск решения: пошаговое руководство (команда IRS-MIREA)

Решение — один Docker-образ `metro-lidar:gpu-dev` (ID `a0e1e7e33476`). В нём всё нужное для работы:
ROS 2 Humble, rosbag2 (SQLite3 и MCAP), RViz2, PyTorch 2.11 + CUDA 12.8 со своими библиотеками CUDA, веса
нейросети Д3, Mesa и команда `metro`. С хоста нужны только Docker, драйвер NVIDIA и NVIDIA Container Toolkit.

Всё делается тремя командами:

```bash
cd dist && sha256sum -c metro-lidar-final.parts.sha256 && cat metro-lidar-final.tar.gz.part-* | gunzip | docker load   # один раз
cd ../solution && ./run.sh --data /путь/к/датасету      # терминал 1: самопроверка, детектор, RViz, консоль статуса
./run.sh play doubleT_obstacle                           # терминал 2: проиграть запись
```

Если всё уже установлено, переходите к §3. Ниже `solution/` — каталог решения (там лежат `run.sh` и
`Dockerfile`), `dist/` — каталог с архивом образа.

## 1. Что нужно на компьютере

| Что | Требование | Как проверить |
|---|---|---|
| ОС | Ubuntu 22.04 (проверено и на 24.04) | `lsb_release -d` |
| Docker | любой: `docker.io` из Ubuntu или Docker CE; BuildKit не нужен | `docker info` без ошибок и без `sudo` |
| Драйвер NVIDIA | ≥ 570 (на стенде 580.173 — подходит) | `nvidia-smi` |
| NVIDIA Container Toolkit | любой текущий | `nvidia-ctk --version` |
| Рабочий стол X11 | для окна RViz; без него решение работает, результаты — в терминале | `echo $DISPLAY` → `:0` или `:1` |
| Диск | ≈ 18 ГБ: части архива 4,5 ГБ + образ 13,3 ГБ (ещё 4,5 ГБ, если склеивать архив в отдельный файл) | `df -h` |
| Память | ≥ 8 ГБ RAM, ≥ 1 ГБ свободной памяти GPU | `free -g`, `nvidia-smi` |
| Данные | каталог с записями rosbag2 (папки с `metadata.yaml` и `.db3` / `.mcap`), любая вложенность | — |

Интернет нужен только для установки пакетов (§2) и скачивания архива (§3). Сам запуск работает без сети.

## 2. Подготовка компьютера (один раз)

Если `docker info`, `nvidia-smi` и `nvidia-ctk --version` уже работают без ошибок, этот раздел пропустите.

**2.1. Docker.**

```bash
sudo apt-get update && sudo apt-get install -y docker.io
sudo usermod -aG docker $USER        # затем выйти из сеанса и войти снова (или: newgrp docker)
docker info > /dev/null && echo "docker OK"
```

**2.2. Драйвер NVIDIA.** `nvidia-smi` должна показать видеокарту и `Driver Version: 570` или новее. Если
команды нет или она пишет ошибку:

```bash
sudo apt-get install -y nvidia-driver-580 && sudo reboot
```

**2.3. NVIDIA Container Toolkit** — без него Docker не может дать контейнеру видеокарту (официальная
инструкция NVIDIA для Ubuntu):

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
```

Проверка: `docker info | grep -i runtimes` показывает `nvidia`.

## 3. Получение образа (один раз, 5–10 минут)

**3.1. Скачать архив.** Архив образа (4,48 ГБ, сжат gzip) лежит в релизе
[v3](https://github.com/garabot11/I.MOSCOW/releases/tag/v3) репозитория частями по 95 МБ: 45 файлов
`metro-lidar-final.tar.gz.part-00` … `part-44` и три служебных — `metro-lidar-final.parts.sha256`,
`metro-lidar-final.tar.gz.sha256`, `metro-lidar-final.image.txt`. Их нужно положить в `dist/`.

```bash
mkdir -p dist && cd dist
base=https://github.com/garabot11/I.MOSCOW/releases/download/v3
for f in metro-lidar-final.parts.sha256 metro-lidar-final.tar.gz.sha256 metro-lidar-final.image.txt \
         $(seq -f 'metro-lidar-final.tar.gz.part-%02g' 0 44); do
  curl -fL -C - -O "$base/$f" || echo "НЕ СКАЧАН: $f (повторите команду — докачает)"
done
```

Скачанные файлы докачиваются при повторе (`-C -`). Другие способы: GitHub CLI —
`gh release download v3 --repo garabot11/I.MOSCOW --pattern 'metro-lidar-final*' --dir dist`; или браузер —
скачать все 48 файлов со страницы релиза в папку `dist/`.

**3.2. Проверить и загрузить в Docker.**

```bash
cd dist
sha256sum -c metro-lidar-final.parts.sha256                      # 45 строк «…: OK»
cat metro-lidar-final.tar.gz.part-* | sha256sum                  # должно совпасть с metro-lidar-final.tar.gz.sha256
cat metro-lidar-final.tar.gz.part-* | gunzip | docker load       # 3–5 минут; в конце: Loaded image: metro-lidar:gpu-dev
docker images metro-lidar                                        # metro-lidar  gpu-dev  a0e1e7e33476  …  13.3GB
```

Склеивать архив в отдельный файл не нужно: `cat … | gunzip | docker load` читает части по порядку.

**Альтернатива — собрать образ из исходников** (нужен интернет, около 4,5 ГБ загрузки; видеокарта и
BuildKit не нужны; версии Python-пакетов зафиксированы в `docker/requirements-torch.txt`):

```bash
cd solution && docker build -t metro-lidar:gpu-dev .
```

## 4. Проверка видеокарты в Docker (30 секунд)

Перед первым запуском убедитесь, что контейнер видит видеокарту:

```bash
docker run --rm --gpus all metro-lidar:gpu-dev nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
# пример для стенда: NVIDIA GeForce RTX 4070 Ti SUPER, 580.173.02
docker run --rm --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all metro-lidar:gpu-dev metro doctor --no-bags --no-color | head -2
# metro doctor …   OK 6  WARN 2  FAIL 0   (GPU mode: gpu, …)
#   [OK  ] gpu   NVIDIA GeForce … torch 2.11.0+cu128, CUDA 12.8, … arch list [...], device capability 8.9
```

Вторая команда выполняет пробный расчёт на GPU (умножение матриц и свёртку, как у Д3) и сравнивает его с CPU.
`GPU mode: gpu` — всё в порядке. Две строки WARN здесь ожидаемы: в этой проверке нет дисплея и данных.

* `could not select device driver "" with capabilities: [[gpu]]` — не установлен NVIDIA Container Toolkit (§2.3).
* `[FAIL] gpu` — причина и точная команда исправления написаны в той же строке (см. также §10).

**Видеокарта стенда (RTX 4070 Ti SUPER, архитектура sm_89).** PyTorch в образе собран с ядрами sm_75, sm_80,
sm_86, sm_90, sm_100 и sm_120. Карта sm_89 выполняет ядра sm_86: двоичная совместимость CUDA внутри
поколения 8.x, на ней работают все колёса PyTorch. Своего CUDA-кода у решения нет: только операции PyTorch,
cuBLAS (в нём есть родные ядра sm_89) и cuDNN, поэтому JIT-компиляция на стенде не нужна. Проверить это на
самой карте мы не могли, поэтому пробный расчёт при каждом старте подтверждает работу на месте. Если он не
пройдёт, решение не упадёт: оно перейдёт в аварийный режим CPU и напишет причину (§9).

## 5. Запуск

```bash
cd solution
./run.sh --data /путь/к/датасету
```

`--data` — каталог, внутри которого лежат папки rosbag2 (на любой глубине, например
`dataset/for_hackathon/doubleT_obstacle`). Каталог монтируется только для чтения в `/data`. Несколько
каталогов: `--data A --data B` (тогда они видны как `/data/A`, `/data/B`).

Что делает `run.sh`:

1. Проверяет Docker, драйвер (`nvidia-smi`), NVIDIA Container Toolkit и делает пробный `docker run --gpus all`.
   Если всё работает, контейнер получает видеокарту. Если нет — запускается аварийный режим CPU с
   объяснением причины (§9). Ничего настраивать вручную не нужно.
2. Запускает постоянный контейнер `metro-lidar`: `--network host`, данные в `/data` только для чтения,
   результаты и логи в `~/metro-lidar-results` на хосте. X11 подключается через отдельный файл Xauthority,
   без `xhost +`.
3. Внутри контейнера выполняет самопроверку `doctor`, запускает детектор и RViz и открывает живую консоль
   статуса.

Так выглядит нормальный старт (около 10 секунд):

```
[run.sh] host GPU: NVIDIA GeForce RTX 5060 Laptop GPU, driver 580.178.04
[run.sh] GPU mode: the container gets --gpus all
[run.sh] starting container metro-lidar from metro-lidar:gpu-dev (data, read-only: /home/user/dataset -> /data; results: /home/user/metro-lidar-results -> /results)
[metro] self-check (bags: metro bags; full report: metro doctor, /results/logs/doctor.json) ...
metro doctor 2026-10-20 19:47:32   OK 9  WARN 0  FAIL 0   (GPU mode: gpu, RViz GL: nvidia_prime)
  [OK  ] gpu        NVIDIA GeForce RTX 5060 Laptop GPU (sm_120 via native sm_120 kernels), torch 2.11.0+cu128, CUDA 12.8, …
  [OK  ] display    X server :0 reachable
  [OK  ] graphics   RViz renders on the GPU: NVIDIA GeForce RTX 5060 Laptop GPU/PCIe/SSE2 (PRIME offload)
  …
[metro] waiting for the detector to be ready (models, GPU warm-up) . 8 s
[metro] D3 on: d3_unet.pt (gpu), warm-up 4.1 s
[metro] detector started (mode: gpu, RViz: on, GL nvidia_prime, input: auto, fixed frame: hesai_lidar)
```

Главное в выводе: `GPU mode: gpu`, `FAIL 0`, `D3 on: d3_unet.pt (gpu)` и `detector started (mode: gpu …)`.
Открывается окно RViz. Пока запись не играет, статус — `UNKNOWN` («waiting for a PointCloud2 topic»), так и
должно быть.

Консоль статуса можно закрыть сочетанием Ctrl-C: детектор продолжит работать. Открыть её снова —
`./run.sh console`.

## 6. Проигрывание записей (второй терминал)

```bash
cd solution
./run.sh bags                                   # список записей: путь относительно /data, формат, размер, длительность, топик
./run.sh play doubleT_obstacle                  # по имени папки (достаточно однозначного хвоста пути)
./run.sh play for_hackathon/roundT_doubleT 0.5  # скорость 0.5
./run.sh play falcon/falcon_test_second_dark 1.0 --start-offset 262   # с 262-й секунды записи
```

Клавиши в терминале проигрывания:

| Клавиша | Действие |
|---|---|
| ПРОБЕЛ | пауза / продолжение |
| ↑ / ↓ | скорость +10 % / −10 % |
| → | следующее облако (на паузе) |
| Ctrl-C | остановить проигрывание |

* **Смена записи без перезапуска.** Просто выполните `./run.sh play <другая запись>`: предыдущая
  остановится сама, детектор увидит скачок времени и сбросит состояние.
* **Любой топик и лидар.** Проигрывается только топик облака точек; он публикуется в `/lidar_points`, и
  детектор находит его сам. Оси датчика (Hesai, Seyond Falcon Prime) и `frame_id` определяются
  автоматически, RViz показывает любую запись без перенастройки.
* **Остановить всё:** `./run.sh stop` — останавливает и удаляет контейнер.

## 7. Что видно в RViz и в консоли

| В RViz | Что это |
|---|---|
| Облако точек (цвет — координата по оси, от синего к зелёному и фиолетовому) | текущий кадр лидара |
| Зелёная линия | ось пути, построенная детектором |
| Синие линии и полупрозрачный синий объём | границы габарита поезда (низ, бока, верх) — коридор, внутри которого ищутся препятствия |
| Красная коробка с подписью `OBSTACLE 56.1 m` и красные точки | подтверждённое препятствие и его отражения |
| Оранжевая полупрозрачная коробка `candidate` | кандидат, ещё не подтверждённый (ждёт второго детектора или следующего кадра) |
| Большая надпись `CLEAR   track checked to 170 m` (зелёная) | препятствий нет на проверенном участке пути до 170 м |
| `OBSTACLE  56.1 m   track checked to 170 m` (красная) | препятствие, расстояние от лидара до его ближайшей поверхности |
| `UNKNOWN (причина)` (жёлтая) | нет данных или не видно полотна пути: нет облаков, поток прервался, всё закрыто |
| Красная строка `EMERGENCY CPU MODE …` | аварийный режим без видеокарты (§9) |

Консоль: одна строка на обработанное облако.

```
t+  225.4s frame   291 | OBSTACLE | distance   54.7 m | checked to 120 m | proc  61.3 ms | nearest object: 54.7 m, 0.5x1.5 m, 41 pts, conf 1.00
```

* `OBSTACLE` / `CLEAR` / `UNKNOWN` — статус; `distance` — расстояние до ближайшего подтверждённого
  препятствия от лидара.
* `checked to N m` — до какой дальности в этом кадре построен и проверен коридор габарита. `CLEAR` относится
  только к этому участку: дальше реальных отражений тоннеля нет или коридор закрыт поворотом.
* `proc` — время обработки кадра. При 10 Гц на кадр есть 100 мс; если кадр обрабатывается дольше, следующий
  кадр заменяет ожидающий (политика «самый свежий кадр»), детектор не отстаёт от потока.
* В аварийном режиме в конце строки стоит `CPU_EMERGENCY`.

Выход для других программ — топик `/obstacles/status` (`metro_obstacle_msgs/ObstacleStatus`): статус,
расстояние, `verified_to_m`, режим `gpu` / `cpu_emergency` / `cpu_forced` и список объектов. Также
`/obstacles/markers` (RViz) и `/obstacles/candidates` (точки объектов). Поля описаны в `solution/README.md`.

## 8. Сценарий показа: что проиграть и что должно получиться

Скорость 1.0 (реальное время), GPU. Ожидаемые результаты — из прогонов образа поставки и демо-видео
(`solution/video/`).

| Команда | Что должно получиться |
|---|---|
| `./run.sh play doubleT_obstacle` | `OBSTACLE` ≈ 56 м с первой секунды, пока объект в габарите (около 6 с), затем `CLEAR`: объект уходит с пути |
| `./run.sh play roundT_doubleT` | поворот и переход в двухпутный тоннель: всё время `CLEAR`, коридор следует за поворотом |
| `./run.sh play squareT_platform_squareT_switch` | станция и стрелка в движении: всё время `CLEAR` |
| `./run.sh play extra/cloud_with_fake_obj 1.0 --start-offset 8` | стена 2 × 2 м с 76 м до 3 м (≈ 4–15 с), кубы 0,3 м с 37–52 м (≈ 26–46 с), низкая плита с ≈ 91 м (≈ 61–66 с); после 80-й секунды записи — `CLEAR` |
| `./run.sh play synthetic_platform_cloud_with_markers 1.0 --start-offset 11` | синтетика организаторов: человек с ≈ 119 м до 8 м |
| `./run.sh play synthetic_switch_cloud_with_markers 1.0 --start-offset 30` | куб на рельсе с ≈ 29 м, подвесной кабель с ≈ 50 м |
| `./run.sh play synthetic_tunnels_cloud_with_markers 1.0 --start-offset 37` | большой куб с ≈ 60 м, балка поперёк рельсов с ≈ 74 м |
| `./run.sh play falcon/falcon_test_second_dark 1.0 --start-offset 262` | второй лидар (Seyond Falcon Prime): тёмный человек — с ≈ 154 м (≈ 11 с), подходит до 100 м, стоит |

Если записи лежат не так, как у нас, имена из первой колонки `./run.sh bags` подходят всегда.

## 9. Режимы GPU и CPU

| Режим | Когда | Что видно |
|---|---|---|
| `gpu` | видеокарта доступна и прошла пробный расчёт | `GPU mode: gpu`, `D3 on: … (gpu)` |
| `cpu_emergency` | видеокарта нужна, но недоступна или сломалась во время работы | баннер на русском и английском при старте и каждые 5 минут, красная строка в RViz, файл `ВНИМАНИЕ_GPU.txt` в папке результатов, `CPU_EMERGENCY` в консоли |
| `cpu_forced` | запуск с `./run.sh --cpu` | баннер один раз |

В аварийном режиме решение продолжает работать: детекторы Д1 и Д2 всегда считаются на CPU, нейросеть Д3
считается каждый третий кадр. С видеокартой Д3 работает в каждом кадре и даёт дополнительные свидетельства
(на Falcon — больше кадров дальнего прохода). Баннер называет причину, точные команды исправления и
контакт команды IRS-MIREA.

Вернуться в режим GPU: устранить причину по баннеру, затем `./run.sh stop && ./run.sh --data …`.

Если видеокарта отказала во время работы (ошибка CUDA), детектор переключается на CPU на ходу, без
перезапуска, и выводит то же предупреждение.

## 10. Если что-то пошло не так

| Симптом | Причина | Что сделать |
|---|---|---|
| `docker is not installed` | нет Docker | §2.1 |
| `cannot talk to the Docker daemon` | демон не запущен или пользователь не в группе `docker` | `sudo systemctl start docker`; `sudo usermod -aG docker $USER`, перелогиниться |
| `image metro-lidar:gpu-dev not found` | образ не загружен | §3.2 |
| `GPU NOT available (no_toolkit …)` | нет NVIDIA Container Toolkit | §2.3, затем `./run.sh stop && ./run.sh --data …` |
| `GPU NOT available (no_host_gpu …)` | `nvidia-smi` не работает на хосте | §2.2 |
| `GPU NOT available (old_driver …)` | драйвер старше 570 | `sudo apt-get install -y nvidia-driver-580 && sudo reboot` |
| `cpu_emergency: CUDA cannot be initialised` / `The GPU test computation failed` | сбой драйвера или GPU | `nvidia-smi`, `sudo dmesg \| grep -i -E 'NVRM\|Xid'`, перезапуск `./run.sh stop && ./run.sh --data …` |
| `cpu_emergency: Not enough free GPU memory` | видеокарту заняли другие программы (нужно ≥ 512 МБ) | закрыть их (`nvidia-smi` покажет какие), перезапустить |
| `no DISPLAY on the host: starting without RViz` | запуск не из графического сеанса (ssh) | запустить из терминала рабочего стола; без RViz всё работает, статус — в консоли |
| `no X cookie for :0 … RViz may not connect` / окно RViz не открылось | нет `xauth` или чужой сеанс X | `sudo apt-get install -y xauth` или `xhost +si:localuser:$(id -un)`, затем `./run.sh stop && ./run.sh --data …` |
| RViz открыт, но медленный (`RViz GL: software`) | OpenGL видеокарты недоступен в контейнере | работает и так (программная отрисовка нагружает CPU); детектор от этого не зависит |
| `bag '…' not found` | неверное имя записи | `./run.sh bags` и имя из первой колонки |
| `cannot be played: no metadata.yaml` | у записи нет `metadata.yaml` | скопировать запись и выполнить `ros2 bag reindex <копия> -s sqlite3` |
| `the container is not running` | контейнер остановлен | `./run.sh --data …` |
| `a container named metro-lidar exists and was not started by run.sh` | конфликт имён | `docker rm -f metro-lidar` |
| Статус `UNKNOWN (no input …)` при проигрывании вне контейнера | общая память Fast DDS не работает между хостом и контейнером | `./run.sh stop && ./run.sh --udp --data …` (§12.3) |
| `UNKNOWN (found several PointCloud2 topics …)` | в той же сети ROS публикуют облака другие устройства | `./run.sh --topic /нужный_топик --data …` или свободный домен: `ROS_DOMAIN_ID=42 ./run.sh --data …` |

Полный отчёт самопроверки — `./run.sh doctor`. Логи — в `~/metro-lidar-results/logs` (§12.6). В консоль
решение не выводит трассировок Python: каждая ошибка — одна понятная строка с исправлением.

## 11. Чек-лист перед показом

1. `nvidia-smi` показывает видеокарту, драйвер ≥ 570.
2. `docker images metro-lidar` показывает `gpu-dev` с ID `a0e1e7e33476`.
3. `docker run --rm --gpus all metro-lidar:gpu-dev nvidia-smi` работает (§4).
4. `./run.sh --data /путь/к/датасету` из терминала рабочего стола: `GPU mode: gpu`, `FAIL 0`, `D3 on … (gpu)`.
5. Окно RViz открылось, в консоли — `UNKNOWN (waiting for a PointCloud2 topic)`.
6. `./run.sh bags` перечисляет нужные записи без `BROKEN`.
7. `./run.sh play doubleT_obstacle` — в RViz `OBSTACLE ≈ 56 m`.
8. Ничего тяжёлого на тех же ядрах CPU не запущено: посторонняя нагрузка замедляет детектор в 1,5–2 раза.
9. Для следующей записи — просто `./run.sh play <имя>`, перезапуск не нужен.
10. В конце — `./run.sh stop`.

## 12. Справочник

### 12.1. Параметры `run.sh`

| Параметр | Что делает |
|---|---|
| `--data DIR` | каталог с записями; можно повторить: каждая папка монтируется в `/data/<имя>` |
| `--results DIR` | каталог результатов и логов (по умолчанию `~/metro-lidar-results`) |
| `--cpu` | принудительный режим CPU |
| `--no-gui` | без RViz |
| `--udp` | Fast DDS только по UDP — если запись проигрывается **вне** контейнера (§12.3) |
| `--topic T` | топик облака для детектора, если запись играется вне контейнера. По умолчанию `auto`: единственный топик PointCloud2 с публикатором, а при нескольких — `/lidar_points` |
| `--cpuset-cpus 0-7` | ограничить контейнер восемью ядрами |
| `--image IMG` | другой образ |
| `--detach` | не открывать консоль статуса |

Команды: `./run.sh play | bags | status | console | doctor [--full] | banner | shell | stop`.
`RVIZ_GEOMETRY=1500x1000+20+40 ./run.sh …` задаёт размер и положение окна RViz.

### 12.2. Команды внутри контейнера

`./run.sh shell` открывает bash в контейнере (окружение ROS уже подключено). Там доступны те же команды
`metro start | play | bags | status | console | doctor | banner | stop` и все инструменты ROS 2:
`ros2 topic list`, `ros2 topic echo /obstacles/status`, `ros2 topic hz /obstacles/status`.

### 12.3. Проигрывание вне контейнера

Если запись проигрывается своим `ros2 bag play` на хосте или в другом контейнере:

```bash
./run.sh --udp --data /путь        # Fast DDS только по UDP: общая память между контейнером и хостом не работает
ros2 bag play /путь/к/записи        # на хосте, тот же ROS_DOMAIN_ID
```

`ROS_DOMAIN_ID` берётся с хоста (по умолчанию 0), `ROS_LOCALHOST_ONLY` не устанавливается. Детектор
подписывается на единственный топик PointCloud2 с публикатором. Если в домене есть другие публикаторы облаков,
он берёт `/lidar_points`, а если `/lidar_points` нет — пишет `UNKNOWN` с подсказкой: тогда топик задаётся
явно (`--topic`). Подписка ведётся с QoS `reliable` (как в записях); для датчика, публикующего best-effort,
есть параметр запуска `qos_reliability:=best_effort`.

### 12.4. Запуск без `run.sh`

```bash
mkdir -p results      # иначе Docker создаст её от root, и логи не запишутся
docker run -d --name metro-lidar --init --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all \
  --network host --shm-size=1g -v /путь/к/датасету:/data:ro -v "$PWD/results":/results \
  metro-lidar:gpu-dev sleep infinity
docker exec -it metro-lidar metro start --no-rviz        # самопроверка, детектор, консоль
docker exec -it metro-lidar metro play doubleT_obstacle  # второй терминал
docker rm -f metro-lidar                                 # остановить
```

Без `--gpus all` тот же запуск работает в аварийном режиме CPU с баннером.

### 12.5. Обработка всей записи без ROS-потока (каждый кадр)

```bash
docker run --rm --gpus all -v /путь/к/датасету:/data:ro -v "$PWD/results":/results metro-lidar:gpu-dev \
  ros2 run metro_obstacle_detector offline --bag /data/for_hackathon/doubleT_obstacle --out /results/doubleT_obstacle.jsonl
```

Строка на кадр и сводка; `--device cpu` и `--no-d3` выбирают устройство и отключают нейросеть.

### 12.6. Самопроверка и логи

`./run.sh doctor` проверяет: GPU (режим, устройство, архитектура, версии torch / CUDA / драйвера, свободная
память), X11 и OpenGL, DDS, `/dev/shm`, число CPU, RAM, диск, ПО и веса моделей, каждую запись под `/data`
(формат, `metadata.yaml`, первое облако, число сообщений). Каждая запись проверяется в отдельном процессе:
битый файл не повесит проверку. Отчёт — `logs/doctor.json`.

Логи в `~/metro-lidar-results/logs`: `detector.log` (журнал узла), `detector.jsonl` (строка на кадр:
статус, расстояние, время), `status_monitor.log`, `play.log`, `doctor.json`, `device.json`, `bag_index.json`.

### 12.7. Проверка запуска (образ `a0e1e7e33476`, 20.10.2026)

`docker/failure_matrix.sh` прогоняет все сценарии через `run.sh` и `metro` и проверяет каждый журнал на
трассировки Python. Результат — **17 из 17 PASS, 0 трассировок**:

| Сценарий | Результат |
|---|---|
| GPU | `gpu`, Д3 на GPU; `doubleT_platform` ×2: 322 кадра, 0 обнаружений, p95 55 мс |
| Нет NVIDIA Container Toolkit | `cpu_emergency`, баннер с командами установки, `ВНИМАНИЕ_GPU.txt`; `roundT_doubleT` ×2: 211 кадров, 0 обнаружений, p95 77 мс |
| `docker run` без `--gpus` | `cpu_emergency`: «started without GPU access» |
| `--cpu` | `cpu_forced`, файла нет |
| Нет DISPLAY | RViz не запускается, детектор работает |
| Запись не существует / обрезана / без `metadata.yaml` / пустая папка | понятное сообщение с исправлением, код возврата 1 |
| `metro bags` по двум корням | 20 записей, битые помечены `BROKEN` |
| Два топика облаков (SQLite3, MCAP), MCAP | 27 / 25 / 28 из 30 кадров |
| Смена записи на ходу (платформа → препятствие 921 тыс. → Falcon → roundT) | 468 кадров, 3 сброса, три разных `frame_id` |
| 8 ядер (`--cpuset-cpus 0-7`) | `squareT_platform_squareT_switch`: 608 кадров за 62 с, p50 20 мс, p95 62 мс, ни одного кадра дольше 100 мс |
| GPU скрыт (`CUDA_VISIBLE_DEVICES=`) | `cpu_emergency`: «CUDA cannot be initialised» |
| Сбой GPU во время работы | `gpu` → `cpu_emergency` на ходу, предупреждение, файл |

Дополнительно: живой поток `doubleT_obstacle` на GPU — 196 из 201 кадра, препятствие с кадра 8 на 55,7 м,
`/obstacles/status` 10,05 Гц, p95 44 мс; запуск с RViz на GPU (PRIME) — `doctor OK 9 WARN 0 FAIL 0`;
тесты в образе с GPU — 255 прошли, 17 пропущено.

---

## English summary

* **Image:** `metro-lidar:gpu-dev`, ID `a0e1e7e33476` — ROS 2 Humble, rosbag2 (SQLite3 + MCAP), RViz2,
  PyTorch 2.11 + CUDA 12.8, the D3 weights, Mesa, the `metro` CLI. The host needs Docker, an NVIDIA driver
  ≥ 570 and the NVIDIA Container Toolkit (install commands in §2.3).
* **Get it:** download the 48 `metro-lidar-final*` files of release
  [v3](https://github.com/garabot11/I.MOSCOW/releases/tag/v3) into `dist/` (the `curl` loop of §3.1, or
  `gh release download v3 --repo garabot11/I.MOSCOW --pattern 'metro-lidar-final*' --dir dist`), then
  `sha256sum -c metro-lidar-final.parts.sha256 && cat metro-lidar-final.tar.gz.part-* | gunzip | docker load`.
  Alternative: `docker build -t metro-lidar:gpu-dev solution` (internet, no GPU, no BuildKit).
* **Check the GPU in Docker:** `docker run --rm --gpus all metro-lidar:gpu-dev nvidia-smi` and
  `docker run --rm --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all metro-lidar:gpu-dev metro doctor --no-bags`
  (`GPU mode: gpu`). The stand's RTX 4070 Ti SUPER (sm_89) runs the sm_86 kernels by CUDA binary
  compatibility; the start-up probe (matmul + conv2d against the CPU) confirms it on site.
* **Run:** `cd solution && ./run.sh --data <dataset dir>` — GPU detection, self-check, detector, RViz and the
  live console (Ctrl-C detaches). Then `./run.sh play <bag> [rate] [--start-offset S]` in a second terminal:
  SPACE pauses, UP/DOWN change the rate; another `play` switches the bag without a restart. Topic, frame and
  sensor axes are detected automatically. `./run.sh stop` ends everything.
* **RViz:** green line = track centre, blue lines / volume = train clearance, red box = confirmed obstacle,
  orange box = unconfirmed candidate, big text = `CLEAR` / `OBSTACLE <distance>` with `track checked to N m`.
* **Emergency CPU mode:** without a usable GPU the solution runs on the CPU (D3 every third frame) and says so:
  RU + EN banner at start and every 5 minutes with the reason, the exact fix and the contact of team
  IRS-MIREA, a red line in RViz, `ВНИМАНИЕ_GPU.txt`, `CPU_EMERGENCY` in the console. A GPU failure
  at run time switches to the CPU on the fly.
* **Troubleshooting:** §10 maps every message to its fix; the console never shows Python tracebacks.
* **Launch check of the delivery image (2026-10-20):** 17 of 17 scenarios PASS with 0 tracebacks; live 10 Hz on
  8 cores p95 62 ms with no frame over 100 ms; 255 tests pass in the image on the GPU.
