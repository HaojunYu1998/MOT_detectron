import copy
import pickle

import numpy as np
import torch
import torch.utils.data
from detectron2.data import detection_utils as d2utils
from detectron2.data import transforms as d2trans
from detectron2.data.common import DatasetFromList, MapDataset


DEBUG = True

class COCO_Style_DataMapper:
    """
    A callable which takes a dataset dict in Ultrasound Dataset format,
    and map it into a format used by the model.
    """

    def __init__(self, cfg, is_train=True):
        self.img_format = cfg.INPUT.FORMAT
        self.augmentations = d2utils.build_augmentation(cfg, is_train)
        self.is_train = is_train

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        has_annotation = "annotations" in dataset_dict

        image = d2utils.read_image(dataset_dict["file_name"], format=self.img_format)
        d2utils.check_image_size(dataset_dict, image)

        # if DEBUG:
        #     aug_input = d2trans.StandardAugInput(image)
        #     transforms = aug_input.apply_augmentations(self.augmentations)
        
        image_shape = image.shape[:2]

        dataset_dict['image'] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1).astype("float32"))
        )

        # if not self.is_train:
        #     for i in range(len(dataset_dict)):
        #         dataset_dict.pop("annotations", None)
        #     return dataset_dict

        if self.is_train:
            assert "annotations" in  dataset_dict
        # Apply transform to annotations
        if has_annotation:
            annos = dataset_dict["annotations"]
            # if DEBUG:
            #     annos = [
            #         d2utils.transform_instance_annotations(
            #             obj, transforms, image_shape,
            #         )
            #         for obj in dataset_dict["annotations"]
            #     ]
            instances = d2utils.annotations_to_instances(annos, image_shape)
            dataset_dict["instances"] = d2utils.filter_empty_instances(instances)

        return dataset_dict
