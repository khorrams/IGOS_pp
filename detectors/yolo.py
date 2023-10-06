import torch
from .yolo_utils.models import Darknet
from .yolo_utils.yolo_fix import YOLOv3_fix

def yolov3spp():
    img_size = 512  
    cfg = "./detectors/yolo_utils/yolov3-spp.cfg"  
    weights = "./weight/yolov3-spp-ultralytics-512.pt" 

    model = Darknet(cfg, img_size)
    weights_dict = torch.load(weights, map_location='cpu')
    model.load_state_dict(weights_dict['model'],strict=False)
    return model

def yolov3spp_fix(labels, feature_index):
    yolo_fix = YOLOv3_fix(
                labels = labels, 
                feature_index = feature_index
                )
    yolo_fix = yolo_fix.to('cuda')
    yolo_fix.eval()
    return yolo_fix