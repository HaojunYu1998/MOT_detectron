from detectron2.config import CfgNode as CN


def add_yolo_config(cfg):
    """
    Add config for DeFCN
    """
    cfg.MODEL.YOLOX = CN()
    cfg.MODEL.YOLOX.DEPTH = 0.33
    cfg.MODEL.YOLOX.WIDTH = 0.25

    cfg.MODEL.DARKNET = CN()
    cfg.MODEL.DARKNET.DEPTH = 21 # or 53
    cfg.MODEL.DARKNET.STEM_OUT_CHANNELS = 32
    cfg.MODEL.DARKNET.OUT_FEATURES = ["dark3", "dark4", "dark5"]

    cfg.MODEL.SCPDARKNET = CN()
    cfg.MODEL.SCPDARKNET.OUT_FEATURES = ["dark3", "dark4", "dark5"]
    cfg.MODEL.SCPDARKNET.DEPTHWISE = False

    cfg.MODEL.YOLOPAFPN = CN()
    cfg.MODEL.YOLOPAFPN.IN_FEATURES = ["dark3", "dark4", "dark5"]
    cfg.MODEL.YOLOPAFPN.IN_CHANNELS = [256, 512, 1024]
    cfg.MODEL.YOLOPAFPN.OUT_FEATURES = ["p3", "p4", "p5"]

    cfg.MODEL.ROI_YOLO_HEAD = CN()
    cfg.MODEL.ROI_YOLO_HEAD.NUM_CLASSES = 1
    cfg.MODEL.ROI_YOLO_HEAD.STRIDES = [8, 16, 32]
    cfg.MODEL.ROI_YOLO_HEAD.IN_FEATURES = ["p3", "p4", "p5"]
    cfg.MODEL.ROI_YOLO_HEAD.DEPTHWISE = False

    # solver
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.LR_MULTIPLIER_NAME = ()
    cfg.SOLVER.LR_MULTIPLIER_VALUE = ()
    cfg.SOLVER.WEIGHT_DECAY = 0.05
    cfg.SOLVER.ADAM_BETA = (0.9, 0.999)


    
    