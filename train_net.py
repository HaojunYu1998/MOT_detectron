"""
Lesion detection for ultrasound video Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

import os
import logging
import time
import torch

import numpy as np
from datetime import datetime
import itertools

from collections import OrderedDict
from typing import Any, Dict, List, Set
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import DatasetEvaluators, DatasetEvaluator, print_csv_format
from detectron2.utils.logger import setup_logger
from detectron2.utils.comm import is_main_process
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.evaluation import COCOEvaluator
from detectron2.data.build import build_detection_train_loader, build_detection_test_loader

from yolox.config import add_yolo_config
from yolox.modeling.utils import backup_code
from yolox.data import COCO_Style_DataMapper


class Trainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(
            dataset_name, tasks=("bbox",), distributed=True, output_dir=output_folder
        )
    
    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        mapper = COCO_Style_DataMapper(cfg, True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        mapper = COCO_Style_DataMapper(cfg, False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_optimizer(cls, cfg, model):
        logger = logging.getLogger(__name__)
        optimizer_type = cfg.SOLVER.get("OPTIMIZER", "SGD")
        if is_main_process():
            print(f"Using optimizer {optimizer_type}")
        if optimizer_type == "SGD":
            optimizer = super().build_optimizer(cfg, model)
            return optimizer

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        lr_multiplier = {k: v for k, v in zip(
            cfg.SOLVER.LR_MULTIPLIER_NAME, cfg.SOLVER.LR_MULTIPLIER_VALUE
        )}
        for key, value in model.named_parameters(recurse=True):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            for name in lr_multiplier:
                if name in key:
                    lr = lr * lr_multiplier[name]
                    if is_main_process():
                        logger.info(f"Learning Rate: {key} {lr}")
                    break
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        if optimizer_type.upper() == "ADAMW":
            optimizer = torch.optim.AdamW(
                params, cfg.SOLVER.BASE_LR, betas=cfg.SOLVER.ADAM_BETA,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_yolo_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(),
                 name="MOT", abbrev_name="mot")
    return cfg


def main(args):
    cfg = setup(args)
    output_dir = cfg.OUTPUT_DIR
    if is_main_process():
        hash_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_code(os.path.abspath(os.path.curdir), os.path.join(output_dir, "code_" + hash_tag))
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
