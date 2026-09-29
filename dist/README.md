# dist

Готовый Docker-образ `metro-lidar:final` (ID `sha256:69acee4e…`) в виде архива
`metro-lidar-final.tar.gz` (462 МБ). GitHub не принимает файлы больше 100 МБ, поэтому в
репозитории архив разрезан на части `metro-lidar-final.tar.gz.part-00` … `part-04`.

Сборка архива, проверка и загрузка образа (из этой папки):

```bash
sha256sum -c metro-lidar-final.parts.sha256                        # проверка частей
cat metro-lidar-final.tar.gz.part-* > metro-lidar-final.tar.gz     # склейка
sha256sum -c metro-lidar-final.tar.gz.sha256                       # 1c05211b… — исходный архив
docker load -i metro-lidar-final.tar.gz                            # → metro-lidar:final
```

Образ можно собрать и без архива, из `../solution/Dockerfile` (см. `../solution/README.md`).
