# simple-person-segmentation

## Preinstalls:

1. Nvidia driver
2. Docker
3. Nvidia docker (nvidia-cuda-toolkit)

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