python train_net.py --num-gpus 4 --eval-only \
--config-file configs/YOLOX/yolox_nano_50000_iters.yaml \
OUTPUT_DIR outputs/yolox_nano_50000_iters \
MODEL.WEIGHTS outputs/yolox_nano_50000_iters/model_final.pth