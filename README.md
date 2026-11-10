# RecSys MVP: рекомендации фильмов


## Запуск в Windows

Нужен Python 3.12. Откройте PowerShell в корне проекта:

```powershell
py -3.12 -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
./.venv/Scripts/python.exe main.py
```

Если окружение уже настроено, достаточно последней команды.
Датасет автоматически скачивается с GroupLens при первом запуске.
Скрипт сравнит пять моделей, выведет метрики и по 10 рекомендаций для пользователя 1.
Веса Two-Tower загружаются автоматически.

## Основные команды

```powershell
# Быстрая проверка простого алгоритма
./.venv/Scripts/python.exe main.py --model popular

# Рекомендации и метрики одной модели
./.venv/Scripts/python.exe main.py --model two-tower --user 10

# Повторное обучение Two-Tower и полное сравнение
./.venv/Scripts/python.exe main.py --train --epochs 18

# Проверка на val; нейросеть обучается только на train, test-веса не используются
./.venv/Scripts/python.exe main.py --stage val --model two-tower --epochs 18
```

Обучение работает на CUDA/MPS при наличии, иначе на CPU. На CPU оно может быть долгим.
Если весов нет, Two-Tower обучится при первом выбранном запуске.
Размер векторов и архитектура фиксированы; --epochs меняет число эпох.

## Файлы

- main.py — обучение, метрики и пример выдачи.
- src/recsys/data.py - данные, признаки и train/val/test.
- src/recsys/base.py - общий интерфейс и исключение известных фильмов.
- src/recsys/baselines.py - две эвристики.
- src/recsys/classic.py - ContentBased и ItemKNN.
- src/recsys/two_tower.py - обучение и загрузка нейросети.
- src/recsys/metrics.py - NDCG, Recall, HitRate, Coverage.
- artifacts/two_tower.pt - исходные обученные веса, совместимые с MVP.
- results/reference_metrics.csv - сохранённые цифры исходного исследования.

Новые результаты записываются отдельно: полное сравнение - results/metrics_test.csv,
одна модель - results/metrics_test_<имя>.csv. Валидация сохраняется с суффиксом val.
Рядом сохраняется JSON с параметрами запуска.
Старые reference_metrics.csv не выдаются за новый прогон.

Сравнение использует все фильмы каталога, исключает уже оценённые и считает метрики на top-10.
batch_ms_per_user - среднее время пакетной выдачи
Данные - оценки фильмов, а не журнал просмотров.
