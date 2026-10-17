# MedAudioLab

Профессиональное приложение для записи, хранения и глубокого спектрального анализа звуковых и ультразвуковых сигналов (частоты дискретизации до 768 кГц). Версия 2 построена на PySide6 и pyqtgraph; подробная архитектура описана в [`project_architecture.md`](project_architecture.md).

## Структура проекта

Код разделён на слои. Зависимости направлены только вниз: `ui` → `services` → `core` / `infrastructure`. Слой `core` ничего не знает ни о Qt, ни об оборудовании.

```text
app/
├── constants.py              # Константы и перечисления (частоты, форматы, фильтры)
├── resource_path.py          # Поиск ресурсов в режиме разработки и в собранном .exe
├── core/                     # Слой 1: чистая предметная логика
│   ├── dsp/                  # FFT, STFT, CWT (Морле), фильтры, огибающая, импульсы, метрики
│   ├── models/               # Неизменяемые модели: AudioBuffer, SignalMetrics, CwtResult, ...
│   └── interfaces/           # Абстрактные интерфейсы (аудио, COM-порт, хранилище, экспорт)
├── infrastructure/           # Слой 2: оборудование и хранение
│   ├── audio/                # sounddevice: захват, воспроизведение, кольцевой буфер
│   ├── serial/               # pyserial: чтение бинарных и текстовых потоков (STM32 и др.)
│   └── storage/              # WAV/CSV/TXT, SQLite, экспорт отчётов и графиков
├── services/                 # Слой 3: сценарии работы приложения
│   ├── recording_service.py  # запись, живой мониторинг, сохранение
│   ├── analysis_service.py   # расчёты DSP в фоновых потоках
│   ├── project_manager.py    # записи на диске и в базе данных
│   └── settings_service.py   # настройки
└── ui/                       # Слой 4: интерфейс (PySide6)
    ├── main_window.py        # главное окно и запуск приложения
    ├── viewmodels/           # состояние вкладок (сигналы Qt)
    ├── tabs/                 # «Запись», «Анализ», «База записей»
    ├── widgets/              # графики, индикатор уровня, выбор устройства
    ├── workers/              # фоновые QThread-задачи
    └── dialogs/              # окно настроек

resources/                    # wavelet_scales.txt, стили (light.qss), иконки
tests/                        # модульные тесты (pytest)
scripts/                      # сборка .exe и вспомогательные скрипты
medaudiolab.spec              # настройки PyInstaller
```

База данных и записи по умолчанию хранятся в `Документы\MedAudioLab` (файл `med_audio_lab.sqlite3`).

## Запуск

Из корневой директории проекта:

```bash
python run.py
```

или, что то же самое:

```bash
python -m app
```

## Тесты

```bash
pip install -e ".[dev]"
pytest
```

Тесты DSP и инфраструктуры не требуют PySide6 и оборудования.

## Сборка EXE

Из корня проекта:

```bat
scripts\build_exe.bat
```

Если запускаете именно PowerShell-скрипт напрямую и политика Windows блокирует `.ps1`, используйте:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_exe.ps1
```

Скрипт создаёт `.venv`, ставит зависимости из `requirements.txt` и PyInstaller, собирает `dist\MedAudioLab.exe` по файлу `medaudiolab.spec` и проверяет, что результат создан. Параметры: `-SkipInstall` (не ставить пакеты), `-VenvPath`, `-DistPath`, `-WorkPath`.

## Требования

Python 3.10+ (рекомендуется 3.11+). Библиотеки (см. `requirements.txt`):
`PySide6`, `pyqtgraph`, `matplotlib`, `numpy`, `scipy`, `sounddevice`, `soundfile`, `pyserial`.

## История

Первая версия приложения (прототип на Tkinter, пакет `med_audio_lab`) заменена версией 2 и сохранена в истории git.
