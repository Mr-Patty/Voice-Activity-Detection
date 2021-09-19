# Инструкции по обучению и запуску модели

## Скачивание данных
**Librispeech**
```
wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
wget https://www.openslr.org/resources/12/train-clean-360.tar.gz

tar -xvf train-clean-100.tar.gz
tar -xvf train-clean-360.tar.gz
```
**NonSpeech**
```
wget http://web.cse.ohio-state.edu/pnl/corpus/HuNonspeech/Nonspeech.zip
unzip Nonspeech.zip
```

## Препроцессинг данных
У всех скриптов есть параметры по умолчанию, но для наглядности привожу здесь все входные параметры скриптов
```
python preprocess.py --path_libri LibriSpeech/ --path_nonspeech Nonspeech/
```
path_libri - пусть до датасета librispeech, 

path_nonspeech - путь до данных NonSpeech

## Обучение модели
```
python train.py --path_libri LibriSpeech/ --path_nonspeech Nonspeech/ --checkpoints checkpoints --number 50000 --device cuda --batch_size 128 --epochs 30
```
path_libri - пусть до датасета librispeech, 

path_nonspeech - путь до данных NonSpeech,

checkpoints - путь где будут сохраняться чекпоинты модели

number - количество файлов для обучения (нужно так как все данные могут не поместиться)

## Конвертирование модели в формат ONNX (необязательно)
```
python convert2onnx.py --checkpoint data/vad.pt
```
checkpoint - путь к предобученной модели на pytorch

## Тестирование модели на тестовых данных
```
python test.py --test_path test/ --checkpoint data/vad.pt --number 1000 --device cuda --type torch --threshold 0.7
```
test_path - путь к тестовым данным, которые были получены из шага препроцессинг данных

checkpoint - путь к предобученной модели либо pytorch либо onnx

number - количество файлов для обучения (нужно так как все данные могут не поместиться)

type - тип предобученной модели либо pytorch либо onnx. Возможные варианты: torch, onnx

threshold - threshold предсказания, нужен для вывода отчета классификации

## Запуск модели на одном файле
```
python run.py --test_audio <path2audio> --checkpoint data/vad.pt --device cuda --type torch --threshold 0.7
```
test_audio - путь к аудио файлу

checkpoint - путь к предобученной модели либо pytorch либо onnx

type - тип предобученной модели либо pytorch либо onnx. Возможные варианты: torch, onnx

threshold - threshold предсказания, нужен для вывода отчета классификации