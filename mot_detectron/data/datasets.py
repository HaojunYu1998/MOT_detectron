from detectron2.data import DatasetCatalog, MetadataCatalog
from path import Path
from detectron2.data.datasets.coco import  load_coco_json

ROOT_DIR = Path(__file__).abspath().parent.parent.parent.parent

DATA_ROOT = ROOT_DIR / "datasets" 

DATASETS_TO_BUILD = {
    "MOT17_train": (DATA_ROOT / "MOT17" / "train", DATA_ROOT / "MOT17" / "annotations" / "train_tmp.json"),
    "MOT17_val": (DATA_ROOT / "MOT17" / "val", DATA_ROOT / "MOT17" / "annotations" / "val_tmp.json"),
    "MOT20_train": (DATA_ROOT / "MOT20" / "train", DATA_ROOT / "MOT20" / "annotations" / "train.json"),
    "MOT20_val": (DATA_ROOT / "MOT20" / "val", DATA_ROOT / "MOT20" / "annotations" / "val.json"),
}

THING_CLASSES = ["person"]

for dataset_name, (image_root, json_file) in DATASETS_TO_BUILD.items():
    image_root = str(image_root)
    json_file = str(json_file)
    DatasetCatalog.register(
        dataset_name,
        lambda image_root=image_root, json_file=json_file: load_coco_json(json_file, image_root)
    )
    MetadataCatalog.get(dataset_name).set(
        thing_classes=THING_CLASSES, image_root=image_root, json_file=json_file
    )


if __name__ == "__main__":
    # print(ROOT_DIR)
    # load datasets
    datasets = DatasetCatalog.get("MOT17_train")
    meta = MetadataCatalog.get("MOT17_train")
    print(datasets[0])