
python train_net.py --num-gpus 4 \
--config-file configs/YOLOX/yolox_nano.yaml \
OUTPUT_DIR outputs/yolox_nano

python train_net.py --num-gpus 4 \
--config-file configs/YOLOX/yolox_s.yaml \
OUTPUT_DIR outputs/yolox_s

python train_net.py --num-gpus 4 \
--config-file configs/YOLOX/yolox_m.yaml \
OUTPUT_DIR outputs/yolox_m

python train_net.py --num-gpus 4 \
--config-file configs/YOLOX/yolox_l.yaml \
OUTPUT_DIR outputs/yolox_l

python train_net.py --num-gpus 4 \
--config-file configs/YOLOX/yolox_x.yaml \
OUTPUT_DIR outputs/yolox_x