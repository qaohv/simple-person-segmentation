# simple-person-segmentation

## Preinstalls:

1. [Nvidia driver](https://www.nvidia.ru/Download/index.aspx?lang=ru)
2. [Docker](https://docs.docker.com/engine/install/)
3. [Nvidia docker](https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(Native-GPU-Support))

## Download dataset

Original data:  https://drive.google.com/file/d/1JdHeGkir1v05NRk7Sxwuu-4Nv4xu7n6s/view?usp=sharing    
Cleaned data:  https://drive.google.com/file/d/1CGW7Dd_mX5w8R_jdVO0Ksvkxh3qOzgfa/view?usp=sharing    

## Build docker:
```
docker build -t person-segmentation .
```

## Run container:
```
docker run -it --rm --gpus=all --shm-size 8G -v `pwd`/data:/data -v `pwd`/logs:/logs person-segmentation
```

## Run train:
```
python src/train.py --train-images /data/picksart_persons/train/images/ \
                    --train-masks /data/picksart_persons/train/masks/ \
                    --val-images /data/picksart_persons/val/images/ \
                    --val-masks /data/picksart_persons/val/masks/ \
                    --batch-size 4 \
                    --logdir /logs/exp1/
```

## Eval model:
```
python src/evaluate.py --images /data/picksart_persons/val/images/ \
                       --masks /data/picksart_persons/val/masks/ \
                       --model /path/to/model.pth \
                       --batch-size 4
```

## Convert pytorch model to tensorrt:
```
python src/convert_to_tensorrt.py --input-model /path/to/pytorch-model.pth \
                                  --output-model /path/to/engine.plan
```

## Run speed benchmark:
```
python src/speed_benchmark.py --torch-model /path/to/model.pth \
                              --trt-engine /path/to/ending.plan
```

### Check onnx convertable to trt
```
trtexec --onnx=/path/to/model.onnx --verbose=True --workspace=9000 --shapes=input:1x3x320x320
```
