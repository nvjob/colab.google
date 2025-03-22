#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Настройки для Piper
# Подробная документация: https://github.com/rhasspy/piper/blob/master/TRAINING.md
LANGUAGE = "ru"  # Язык для espeak-ng (например, "ru" для русского, "de" для немецкого, "en-us" для английского)
INPUT_DIR = "/workspace/dataset"  # Директория с исходными данными (должна содержать metadata.csv и wav/)
OUTPUT_DIR = "/workspace/output3"  # Директория для обучения (будет содержать config.json и dataset.jsonl)
SAMPLE_RATE = 22050  # Частота дискретизации (medium quality = 22050 Hz, low = 16000 Hz, high = 22050 Hz с большей моделью)
NUM_SPEAKERS = 3  # Количество спикеров в датасете (drug, egirl, keira)
BATCH_SIZE = 50  # Размер батча (зависит от объема видеопамяти GPU, для 24GB рекомендуется 32)
MAX_EPOCHS = 6000  # Максимальное количество эпох (обычно 2000 для обучения с нуля, 1000 для fine-tuning)
CHECKPOINT_PATH = "/workspace/ukr.ckpt"  # Путь к чекпоинту для fine-tuning в Docker-контейнере
EXPORT_PATH = "/workspace/3voice.onnx"  # Путь для экспорта модели в формате ONNX
MAX_PHONEME_IDS = 800  # Максимальное количество фонемных ID (увеличено для длинных предложений на русском языке)

import os
import subprocess
import sys
import shutil
import importlib.util

# Настройки для оптимизации использования памяти
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Попытка освободить память CUDA перед запуском
try:
    import torch
    torch.cuda.empty_cache()
except ImportError:
    pass

def check_dependencies():
    """Проверка наличия необходимых зависимостей
    
    Piper требует torch и pytorch-lightning для работы.
    Также необходим espeak-ng, который должен быть установлен в системе.
    """
    missing_deps = []
    
    # Проверка torch
    try:
        import torch
        print(f"✓ torch {torch.__version__} установлен")
        
        # Проверка CUDA
        if torch.cuda.is_available():
            print(f"✓ CUDA доступен: {torch.cuda.get_device_name(0)}")
            print(f"  Доступная память GPU: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.2f} GB")
            
            # Проверка свободной памяти
            free_mem = torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024
            print(f"  Зарезервировано памяти: {free_mem:.2f} GB")
            
            # Освобождаем кэш CUDA
            torch.cuda.empty_cache()
            print("  Кэш CUDA очищен")
        else:
            print("⚠ ВНИМАНИЕ: CUDA недоступен. Обучение на CPU будет очень медленным.")
            print("  Убедитесь, что Docker запущен с поддержкой NVIDIA GPU (--gpus all)")
    except ImportError:
        missing_deps.append("torch")
        print("✗ torch не установлен")
    
    # Проверка pytorch_lightning
    try:
        import pytorch_lightning
        print(f"✓ pytorch_lightning {pytorch_lightning.__version__} установлен")
    except ImportError:
        missing_deps.append("pytorch_lightning")
        print("✗ pytorch_lightning не установлен")
    
    # Проверка piper_train
    try:
        # Проверяем, можно ли импортировать модуль piper_train
        import piper_train
        print(f"✓ piper_train установлен")
    except ImportError:
        missing_deps.append("piper_train")
        print("✗ piper_train не установлен")
    
    # Проверка espeak-ng
    try:
        result = subprocess.run(["espeak-ng", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ espeak-ng установлен: {result.stdout.strip()}")
        else:
            print("⚠ espeak-ng установлен, но возникла ошибка при проверке версии")
    except FileNotFoundError:
        print("✗ espeak-ng не установлен")
        missing_deps.append("espeak-ng")
    
    if missing_deps:
        print("\nОтсутствуют необходимые зависимости:")
        for dep in missing_deps:
            print(f"  - {dep}")
        return False
    
    print("\nВсе необходимые зависимости установлены")
    return True

def preprocess_dataset():
    """Подготовка датасета
    
    Этап 1: Подготовка датасета
    
    Создает config.json и dataset.jsonl из исходных данных.
    Исходные данные должны быть в формате:
    - metadata.csv: файл с разделителем | и колонками id|speaker|text
    - wav/: директория с аудиофайлами в формате WAV (например, wav/1234.wav для id=1234)
    
    Результат:
    - config.json: настройки голоса (sample_rate, language, phoneme_map и т.д.)
    - dataset.jsonl: данные для обучения (phoneme_ids, audio_paths и т.д.)
    - аудиофайлы .pt: нормализованные аудиофайлы и спектрограммы
    """
    print(f"Подготовка датасета из {INPUT_DIR} в {OUTPUT_DIR}...")
    
    # Проверяем наличие директории с данными
    if not os.path.exists(INPUT_DIR):
        print(f"ОШИБКА: Директория {INPUT_DIR} не найдена.")
        print(f"Создайте директорию {INPUT_DIR} с файлом metadata.csv и папкой wav/")
        return
    
    # Проверяем наличие metadata.csv
    if not os.path.exists(os.path.join(INPUT_DIR, "metadata.csv")):
        print(f"ОШИБКА: Файл metadata.csv не найден в директории {INPUT_DIR}")
        print("Создайте файл metadata.csv с форматом id|speaker|text")
        return
    
    # Создаем директорию для обучения, если её нет
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    cmd = [
        "python", "-m", "piper_train.preprocess",
        "--language", LANGUAGE,
        "--input-dir", INPUT_DIR,
        "--output-dir", OUTPUT_DIR,
        "--dataset-format", "ljspeech",  # Формат LJSpeech: id|speaker|text
        "--sample-rate", str(SAMPLE_RATE),
        "--max-workers", "4"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("Предобработка завершена")
    except subprocess.CalledProcessError as e:
        print(f"ОШИБКА при предобработке: {e}")
        print("Проверьте, что установлен piper_train и espeak-ng")

def train_model():
    """Обучение модели
    
    Этап 2: Обучение модели голоса
    
    Использует config.json, dataset.jsonl и аудиофайлы .pt для обучения модели.
    Можно обучать с нуля или fine-tune существующую модель (рекомендуется).
    
    Для fine-tuning нужна модель с тем же качеством и sample_rate,
    но не обязательно на том же языке.
    
    Результат:
    - Чекпоинты модели в директории lightning_logs/version_X/checkpoints/
    """
    print("Начало обучения модели...")
    
    # Проверяем наличие предобработанных данных
    if not os.path.exists(os.path.join(OUTPUT_DIR, "config.json")) or not os.path.exists(os.path.join(OUTPUT_DIR, "dataset.jsonl")):
        print(f"ОШИБКА: Файлы config.json или dataset.jsonl не найдены в {OUTPUT_DIR}")
        print("Сначала выполните предобработку датасета (пункт 1)")
        return
    
    # Очищаем кэш CUDA перед запуском
    try:
        import torch
        torch.cuda.empty_cache()
        print("Кэш CUDA очищен перед запуском обучения")
    except:
        pass
    
    cmd = [
        "python", "-m", "piper_train",
        "--dataset-dir", OUTPUT_DIR,  # Директория с config.json и dataset.jsonl
        "--accelerator", "gpu",  # Использовать GPU
        "--devices", "1",  # Количество используемых GPU
        "--batch-size", str(BATCH_SIZE),  # Размер батча (зависит от GPU)
        "--validation-split", "0.0",  # Доля данных для валидации (0.05 = 5%)
        "--num-test-examples", "0",  # Количество примеров для тестирования
        "--max_epochs", str(MAX_EPOCHS),  # Максимальное количество эпох
        "--checkpoint-epochs", "100",  # Сохранять чекпоинт каждую эпоху
        "--precision", "32",  # Точность вычислений (32 или 16)
        "--max-phoneme-ids", str(MAX_PHONEME_IDS)  # Отбрасывать слишком длинные предложения
    ]
    
    # Проверяем наличие чекпоинта для fine-tuning
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Найден чекпоинт для fine-tuning: {CHECKPOINT_PATH}")
        # Для продолжения обучения существующей multi-speaker модели
        cmd.extend([
            "--resume_from_checkpoint", CHECKPOINT_PATH
        ])
    else:
        print(f"ОШИБКА: Чекпоинт {CHECKPOINT_PATH} не найден!")
        print(f"Проверьте, что файл {CHECKPOINT_PATH} существует в текущей директории.")
        return
    
    try:
        # Устанавливаем переменные окружения для оптимизации памяти
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        
        # Запускаем обучение с ограничением использования памяти
        print("Запуск обучения с оптимизацией использования памяти...")
        subprocess.run(cmd, check=True)
        print("Обучение завершено")
    except subprocess.CalledProcessError as e:
        print(f"ОШИБКА при обучении: {e}")
        print("Проверьте наличие GPU и установленных зависимостей")

def export_model():
    """Экспорт модели в ONNX формат
    
    Этап 3: Экспорт модели голоса
    
    Конвертирует обученную модель в формат ONNX для использования в Piper.
    Также копирует config.json для использования с моделью.
    
    Результат:
    - model.onnx: модель в формате ONNX
    - model.onnx.json: конфигурация модели
    """
    print("Экспорт модели в ONNX формат...")
    
    # Проверяем наличие директории с чекпоинтами
    checkpoint_dir = os.path.join(OUTPUT_DIR, "lightning_logs", "version_0", "checkpoints")
    if not os.path.exists(checkpoint_dir):
        print(f"ОШИБКА: Директория с чекпоинтами не найдена: {checkpoint_dir}")
        print("Сначала выполните обучение модели (пункт 2)")
        return
    
    # Находим последний чекпоинт
    checkpoints = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")]
    
    if not checkpoints:
        print("ОШИБКА: Чекпоинты не найдены")
        print("Убедитесь, что обучение модели завершилось успешно")
        return
    
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    print(f"Найден последний чекпоинт: {os.path.basename(latest_checkpoint)}")
    
    cmd = [
        "python", "-m", "piper_train.export_onnx",
        latest_checkpoint,  # Путь к чекпоинту модели
        EXPORT_PATH  # Путь для сохранения ONNX модели
    ]
    
    try:
        subprocess.run(cmd, check=True)
        
        # Копируем config.json для использования с моделью
        shutil.copy(
            os.path.join(OUTPUT_DIR, "config.json"),
            f"{EXPORT_PATH}.json"
        )
        
        print(f"Модель экспортирована в {EXPORT_PATH}")
    except subprocess.CalledProcessError as e:
        print(f"ОШИБКА при экспорте модели: {e}")
    except Exception as e:
        print(f"ОШИБКА при копировании config.json: {e}")

def test_model():
    """Тестирование модели
    
    Использует экспортированную модель для синтеза речи.
    Генерирует аудиофайл test.wav с тестовым предложением.
    
    Для более сложного тестирования можно использовать piper-phonemize
    и piper_train.infer для генерации аудио из фонем.
    """
    print("Тестирование модели...")
    
    # Проверяем наличие экспортированной модели
    if not os.path.exists(EXPORT_PATH) or not os.path.exists(f"{EXPORT_PATH}.json"):
        print(f"ОШИБКА: Файлы модели не найдены: {EXPORT_PATH} или {EXPORT_PATH}.json")
        print("Сначала экспортируйте модель (пункт 3)")
        return
    
    test_text = "Это тестовое предложение."
    test_wav = "/workspace/test.wav"
    
    # По умолчанию используем диктора с ID 0
    speaker_id = 0
    cmd = f'echo "{test_text}" | piper -m {EXPORT_PATH} --output_file {test_wav} --speaker-id {speaker_id}'
    print(f"Используем диктора с ID {speaker_id} для тестирования")
    
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"Тест завершен, аудио сохранено в {test_wav}")
        
        # Тестируем всех дикторов
        print("Тестирование всех дикторов...")
        for speaker_id in range(NUM_SPEAKERS):  # Тестируем всех дикторов
            test_wav = f"/workspace/test_speaker_{speaker_id}.wav"
            cmd = f'echo "{test_text}" | piper -m {EXPORT_PATH} --output_file {test_wav} --speaker-id {speaker_id}'
            print(f"Тестирование диктора с ID {speaker_id}...")
            subprocess.run(cmd, shell=True, check=True)
            print(f"Аудио сохранено в {test_wav}")
    except subprocess.CalledProcessError as e:
        print(f"ОШИБКА при тестировании модели: {e}")
        print("Проверьте, что piper установлен и модель экспортирована корректно")

def main():
    """Основная функция
    
    Предоставляет интерактивное меню для выполнения всех этапов
    обучения голосовой модели Piper:
    1. Подготовка датасета
    2. Обучение модели
    3. Экспорт модели
    4. Тестирование модели
    
    Для полного процесса обучения выполните шаги по порядку.
    """
    print("=" * 80)
    print("Запуск программы для работы с Piper в Docker-контейнере")
    print("=" * 80)
    
    if not check_dependencies():
        input("Нажмите Enter для выхода...")
        return
    
    try:
        while True:
            print("\nВыберите действие:")
            print("1. Подготовить датасет")
            print("2. Обучить модель")
            print("3. Экспортировать модель")
            print("4. Протестировать модель")
            print("0. Выход")
            
            choice = input("Ваш выбор: ")
            
            if choice == "1":
                preprocess_dataset()
            elif choice == "2":
                train_model()
            elif choice == "3":
                export_model()
            elif choice == "4":
                test_model()
            elif choice == "0":
                break
            else:
                print("Неверный выбор")
    except Exception as e:
        print(f"Произошла непредвиденная ошибка: {e}")
        input("Нажмите Enter для выхода...")

if __name__ == "__main__":
    main()
