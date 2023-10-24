# --------------------------------------------------------
# Copied from yolov3_spp_gradcam repo
# https://github.com/yipintiancheng/yolov3_spp_gradcam/tree/master
# We thank the authors for their excellent work.
# --------------------------------------------------------

import torch
from .yolo_utils.models import Darknet
from .yolo_utils.yolo_fix import YOLOv3_fix

def yolov3spp(url = "./weight/yolov3-spp-ultralytics-512.pt"):
    img_size = 512  
    cfg = "./detectors/yolo_utils/yolov3-spp.cfg"  

    model = Darknet(cfg, img_size)
    weights_dict = torch.load(url, map_location='cpu')
    model.load_state_dict(weights_dict['model'],strict=False)
    return model

def yolov3spp_fix(labels, feature_index, url = "./weight/yolov3-spp-ultralytics-512.pt"):
    yolo_fix = YOLOv3_fix(
                labels = labels, 
                feature_index = feature_index,
                url = url
                )
    yolo_fix = yolo_fix.to('cuda')
    yolo_fix.eval()
    return yolo_fix