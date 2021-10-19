#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.


import torch
import torch.nn as nn
from itertools import chain

from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN

from detectron2.modeling import build_roi_heads, build_backbone, META_ARCH_REGISTRY
from detectron2.layers import batched_nms
from detectron2.structures import Instances, Boxes, pairwise_iou
from detectron2.modeling.postprocessing import detector_postprocess

from .utils import imagelist_from_tensors, box_cxcywh_to_xyxy, box_xyxy_to_cxcywh

DEBUG = True


@META_ARCH_REGISTRY.register()
class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """
    def __init__(self, cfg):
        super().__init__()

        self.backbone = build_backbone(cfg)
        self.head = build_roi_heads(cfg, self.backbone.output_shape())

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        self.register_buffer(
            "pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1)
        )
        self.register_buffer(
            "pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1)
        )
        self.nms_thresh = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

        self.to(self.device)

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        # fpn output content features of [dark3, dark4, dark5]
        # (bs, d_model, h, w)
        fpn_outs = self.backbone(images.tensor)

        if self.training:
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                for i in range(len(gt_instances)):
                    gt_instances[i].gt_classes = torch.zeros_like(gt_instances[i].gt_classes).long()
                targets = self.prepare_targets(gt_instances)
                # print([len(gt) for gt in gt_instances])
            else:
                raise AttributeError("Failed to get 'instances' from training image.")
            iou_loss, obj_loss, cls_loss, l1_loss = self.head(
                fpn_outs, targets, images.tensor
            )
            outputs = {
                "obj_loss": obj_loss,
                "cls_loss": cls_loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss
            }
        else:
            # gt_instances = batched_inputs[0]["instances"].to(self.device)
            outputs = self.head(fpn_outs)
            outputs = self.postprocess(outputs, images.image_sizes)
            # anno_boxes = gt_instances.gt_boxes
            # pred_boxes = outputs[0]["instances"].pred_boxes
            # iou_mat = pairwise_iou(pred_boxes, anno_boxes)
            # det_anno_mask = (iou_mat > 0.9).any(dim=0)
            # print(int(det_anno_mask.sum()), len(det_anno_mask), len(pred_boxes))

        return outputs

    def prepare_targets(self, gt_instances):
        """
        Return:
            targets: Tensor(batch, gt_per_img, 5) for (class, cx, cy, w, h)
        """
        num_gt_per_img = [len(gt) for gt in gt_instances]
        max_gt_per_img = max(num_gt_per_img)
        targets = torch.zeros(len(num_gt_per_img), max_gt_per_img, 5).to(self.device)
        for i in range(len(gt_instances)):
            gt_boxes = gt_instances[i].gt_boxes.tensor
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes).reshape(-1, 4)
            target = torch.cat([gt_instances[i].gt_classes.reshape(-1, 1), gt_boxes], dim=-1)
            targets[i, :num_gt_per_img[i], ...] = target
        return targets

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device, non_blocking=True) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = imagelist_from_tensors(images, self.backbone.size_divisibility)
        return images

    def postprocess(self, outputs, image_sizes):
        # TOP 100
        results = []
        for output, image_size in zip(outputs, image_sizes):
            output = output.reshape(-1, output.shape[-1])
            output = output[output[..., 4].topk(1000).indices]
            # NMS
            scores = output[..., 4]
            boxes = box_cxcywh_to_xyxy(output[..., :4])
            idxs = torch.zeros_like(scores).long() + 1
            # keep = batched_nms(boxes, scores, idxs, iou_threshold=self.nms_thresh)
            # scores = scores[keep].reshape(-1)
            # boxes = boxes[keep].reshape(-1, 4)
            # idxs = torch.zeros_like(scores).long() + 1
            # Instances
            # if DEBUG:
            #     if image_size[0] == 1080:
            #         image_size = (750, 1333)
            #     elif image_size[0] == 480:
            #         image_size = (800, 1067)
            #     else:
            #         print(image_size)
            output = Instances(image_size)
            output.pred_boxes = Boxes(boxes)
            output.scores = scores
            output.pred_classes = idxs
            # if DEBUG:
            #     if output.image_size[0] == 750:
            #         # output.image_size = (750, 1333)
            #         output = detector_postprocess(output, 1080, 1920)
            #     elif output.image_size[0] == 800:
            #         # output._image_size = (800, 1067)
            #         output = detector_postprocess(output, 480, 640)
            # h=1080, w=1920, new_h=750, new_w=1333
            # h=480, w=640, new_h=800, new_w=1067
            results.append({"instances": output})
        return results

    def reset(self):
        pass
