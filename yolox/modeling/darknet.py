#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

from torch import nn

from .network_blocks import BaseConv, CSPLayer, DWConv, Focus, ResLayer, SPPBottleneck
from detectron2.modeling import BACKBONE_REGISTRY, Backbone
from detectron2.layers import ShapeSpec


class Darknet(Backbone):
    # number of blocks from dark2 to dark5.
    depth2blocks = {21: [1, 2, 2, 1], 53: [2, 8, 8, 4]}

    def __init__(
        self,
        in_channels=3,
        depth=21,
        stem_out_channels=32,
        out_features=["dark3", "dark4", "dark5"],
    ):
        """
        Args:
            depth (int): depth of darknet used in model, usually use [21, 53] for this param.
            in_channels (int): number of input channels, for example, use 3 for RGB image.
            stem_out_channels (int): number of output chanels of darknet stem.
                It decides channels of darknet layer2 to layer5.
            out_features (Tuple[str]): desired output layer name.
        """
        super().__init__()
        self._out_features = out_features
        assert len(self._out_features)
        self.stem = nn.Sequential(
            BaseConv(in_channels, stem_out_channels, ksize=3, stride=1, act="lrelu"),
            *self.make_group_layer(stem_out_channels, num_blocks=1, stride=2),
        )
        in_channels = stem_out_channels * 2  # 64
        self._out_feature_channels = {"stem": in_channels}

        num_blocks = Darknet.depth2blocks[depth]
        # create darknet with `stem_out_channels` and `num_blocks` layers.
        # to make model structure more clear, we don't use `for` statement in python.
        self.dark2 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[0], stride=2)
        )
        in_channels *= 2  # 128
        self._out_feature_channels["dark2"] = in_channels
        
        self.dark3 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[1], stride=2)
        )
        in_channels *= 2  # 256
        self._out_feature_channels["dark3"] = in_channels

        self.dark4 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[2], stride=2)
        )
        in_channels *= 2  # 512
        self._out_feature_channels["dark4"] = in_channels

        self.dark5 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[3], stride=2),
            *self.make_spp_block([in_channels, in_channels * 2], in_channels * 2),
        )
        self._out_feature_channels["dark5"] = in_channels
        self._out_feature_strides = {"stem": 2, "dark2": 4, "dark3": 8, "dark4": 16, "dark5": 32}

    def make_group_layer(self, in_channels: int, num_blocks: int, stride: int = 1):
        "starts with conv layer then has `num_blocks` `ResLayer`"
        return [
            BaseConv(in_channels, in_channels * 2, ksize=3, stride=stride, act="lrelu"),
            *[(ResLayer(in_channels * 2)) for _ in range(num_blocks)],
        ]

    def make_spp_block(self, filters_list, in_filters):
        m = nn.Sequential(
            *[
                BaseConv(in_filters, filters_list[0], 1, stride=1, act="lrelu"),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                SPPBottleneck(
                    in_channels=filters_list[1],
                    out_channels=filters_list[0],
                    activation="lrelu",
                ),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                BaseConv(filters_list[1], filters_list[0], 1, stride=1, act="lrelu"),
            ]
        )
        return m

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self._out_features}

    def output_shape(self):
        """
        Returns:
            dict[str->ShapeSpec]
        """
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


class CSPDarknet(Backbone):
    def __init__(
        self,
        in_channels=3,
        base_depth=3,
        base_channels=64,
        out_features=["dark3", "dark4", "dark5"],
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        self._out_features = out_features
        assert len(self._out_features)
        Conv = DWConv if depthwise else BaseConv

        # stem
        self.stem = Focus(in_channels, base_channels, ksize=3, act=act)
        self._out_feature_channels = {"stem": base_channels}

        # dark2
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(
                base_channels * 2,
                base_channels * 2,
                n=base_depth,
                depthwise=depthwise,
                act=act,
            ),
        )
        self._out_feature_channels["dark2"] = base_channels * 2

        # dark3
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(
                base_channels * 4,
                base_channels * 4,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )
        self._out_feature_channels["dark3"] = base_channels * 4

        # dark4
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(
                base_channels * 8,
                base_channels * 8,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )
        self._out_feature_channels["dark4"] = base_channels * 8

        # dark5
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(
                base_channels * 16,
                base_channels * 16,
                n=base_depth,
                shortcut=False,
                depthwise=depthwise,
                act=act,
            ),
        )
        self._out_feature_channels["dark5"] = base_channels * 16
        self._out_feature_strides = {"stem": 2, "dark2": 4, "dark3": 8, "dark4": 16, "dark5": 32}
        self._size_divisibility = self._out_feature_strides["dark5"]

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self._out_features}

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
def build_darknet_backbone(cfg, input_shape):
    depth = cfg.MODEL.DARKNET.DEPTH
    stem_out_channels = cfg.MODEL.DARKNET.STEM_OUT_CHANNELS
    out_features = cfg.MODEL.DARKNET.OUT_FEATURES
    return Darknet(
        in_channels=input_shape.channels,
        depth=depth,
        stem_out_channels=stem_out_channels,
        out_features=out_features
    )


@BACKBONE_REGISTRY.register()
def build_scpdarknet_backbone(cfg, input_shape):
    depth_mul = cfg.MODEL.SCPDARKNET.DEPTH_MUL
    width_mul = cfg.MODEL.SCPDARKNET.WIDTH_MUL
    base_depth = max(round(depth_mul * 3), 1)  # 3
    base_channels = int(width_mul * 64)  # 64
    out_features = cfg.MODEL.SCPDARKNET.OUT_FEATURES
    depthwise = cfg.MODEL.SCPDARKNET.DEPTHWISE
    
    return CSPDarknet(
        in_channels=input_shape.channels,
        base_depth=base_depth,
        base_channels=base_channels,
        out_features=out_features,
        depthwise=depthwise,
        act="relu"# "silu"
    )
        