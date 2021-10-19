#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from .darknet import CSPDarknet
from .network_blocks import BaseConv, CSPLayer, DWConv
from .darknet import build_darknet_backbone, build_scpdarknet_backbone

from detectron2.modeling import build_roi_heads, build_backbone, META_ARCH_REGISTRY
from detectron2.modeling.backbone import Backbone, BACKBONE_REGISTRY
from detectron2.layers import ShapeSpec


class YOLOPAFPN(Backbone):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        bottom_up,
        depth=1.0,
        width=1.0,
        in_features=["dark3", "dark4", "dark5"],
        out_features=["p3", "p4", "p5"],
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        self.backbone = bottom_up
        self.in_features = in_features
        self._out_features = out_features
        assert len(self._out_features)
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )
        self._out_feature_channels = {"p3": 256, "p4": 512, "p5": 1024}
        self._out_feature_strides = {"p3": 8, "p4": 16, "p5": 32}
        self._size_divisibility = self._out_feature_strides["p5"]
    
    @property
    def size_divisibility(self):
        return self._size_divisibility

    def forward(self, input):

        #  backbone
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = features

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = [pan_out2, pan_out1, pan_out0]
        return dict(zip(self._out_features, outputs))

    def output_shape(self):
        """
        Returns:
            dict[str->ShapeSpec]
        """
        # this is a backward-compatible default
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


@BACKBONE_REGISTRY.register()
def build_yolopafpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_scpdarknet_backbone(cfg, input_shape)
    depth = cfg.MODEL.YOLOX.DEPTH
    width = cfg.MODEL.YOLOX.WIDTH
    in_features = cfg.MODEL.YOLOPAFPN.IN_FEATURES
    out_features = cfg.MODEL.YOLOPAFPN.OUT_FEATURES
    in_channels = cfg.MODEL.YOLOPAFPN.IN_CHANNELS
    depthwise = cfg.MODEL.SCPDARKNET.DEPTHWISE

    backbone = YOLOPAFPN(
        bottom_up=bottom_up,
        depth=depth,
        width=width,
        in_features=in_features,
        out_features=out_features,
        in_channels=in_channels,
        depthwise=depthwise,
        act="relu"# "silu",
    )
    return backbone