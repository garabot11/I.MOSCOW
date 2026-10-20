# dist

Готовый Docker-образ решения `metro-lidar:gpu-dev` (ID `sha256:a0e1e7e33476…`) в виде архива
`metro-lidar-final.tar.gz` (`docker save | gzip`, 4 476 104 491 байт). Сам архив лежит в
[релизе v3](https://github.com/garabot11/I.MOSCOW/releases/tag/v3) частями по 95 МБ (45 частей); в этой
папке репозитория — контрольные суммы, описание образа и скрипт, которым архив сделан.

| Файл | Где | Что это |
|---|---|---|
| `metro-lidar-final.tar.gz.part-00` … `part-44` | релиз v3 | части архива |
| `metro-lidar-final.parts.sha256` | здесь и в релизе | SHA-256 каждой части |
| `metro-lidar-final.tar.gz.sha256` | здесь и в релизе | SHA-256 целого архива: `ec3f2167ea408a687478fac07656d03dc4ce61ab46cc549fd324cd18ab7c04a3` |
| `metro-lidar-final.image.txt` | здесь и в релизе | тег, ID и размер образа, дата |
| `make_archive.sh` | здесь | как архив сделан |

## Скачивание, проверка и загрузка (из этой папки)

```bash
base=https://github.com/garabot11/I.MOSCOW/releases/download/v3
for f in $(seq -f 'metro-lidar-final.tar.gz.part-%02g' 0 44); do curl -fL -C - -O "$base/$f"; done
sha256sum -c metro-lidar-final.parts.sha256                      # 45 строк «…: OK»
cat metro-lidar-final.tar.gz.part-* | sha256sum                  # = metro-lidar-final.tar.gz.sha256
cat metro-lidar-final.tar.gz.part-* | gunzip | docker load       # Loaded image: metro-lidar:gpu-dev
docker images metro-lidar                                        # gpu-dev  a0e1e7e33476  13.3GB
```

Нужно ≈ 18 ГБ свободного места (части 4,5 ГБ + образ 13,3 ГБ). Затем запуск:
`../solution/run.sh --data <датасет>`; подробно — [`../solution/docs/launch.md`](../solution/docs/launch.md).

## Создание архива

```bash
dist/make_archive.sh metro-lidar:gpu-dev [--force]   # --force заменяет прежние metro-lidar-final.tar.gz*
```

Скрипт выполняет `docker save | gzip -n`, считает SHA-256 и режет архив на части по 95 МБ
(`--part-size`). Затем проверяет, что склейка частей даёт тот же архив, и пишет
`metro-lidar-final.image.txt`. Загрузка проверена: после `docker load` появляется `metro-lidar:gpu-dev` с
ID `a0e1e7e33476`; в образе с GPU проходят 255 тестов, 17 пропущено.
