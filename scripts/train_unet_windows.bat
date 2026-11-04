@echo off
call .venv\Scripts\activate
python train.py --config config.yaml --architecture unet --encoder mobilenet_v2 --output-dir outputs/unet_mobilenetv2
