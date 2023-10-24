# --------------------------------------------------------
# Copied from yolov3_spp_gradcam repo
# https://github.com/yipintiancheng/yolov3_spp_gradcam/tree/master
# We thank the authors for their excellent work.
# --------------------------------------------------------

import torch
from .models import Darknet

import torch
import torch.nn as nn

class YOLOv3_fix(nn.Module):
    def __init__(self,
                 labels = None, 
                 feature_index = None, 
                 url = "./weight/yolov3-spp-ultralytics-512.pt"
                 ):
        super(YOLOv3_fix, self).__init__()
        self.img_size = 512  
        self.cfg = "./detectors/yolo_utils/yolov3-spp.cfg"  
        self.weights = url

        self.model = Darknet(self.cfg, self.img_size)
        self.weights_dict = torch.load(self.weights, map_location='cpu')
        self.model.load_state_dict(self.weights_dict['model'],strict=False)
        self.model=self.model.to('cuda')
        self.labels=labels
        self.module_list=self.model.module_list 
        self.module_defs=self.model.module_defs
        self.feature_index=feature_index


    # ---------------------- Main Process for Training ----------------------
    def forward(self, x):
        output=self.model(x)[0]
        output_score=output[:,self.feature_index:self.feature_index+1, self.labels+5:self.labels+6].clone() * output[:,self.feature_index:self.feature_index+1, 4:5].clone()
        return output_score