@echo off
call .venv\Scripts\activate
python train.py --config config.yaml --architecture deeplabv3plus --encoder resnet50 --batch-size 2 --output-dir outputs/deeplabv3plus_resnet50
