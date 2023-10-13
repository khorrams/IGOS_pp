# --------------------------------------------------------
# Copied from TorchVision repo
# https://github.com/pytorch/vision/tree/main
# We thank the authors for their excellent work.
# --------------------------------------------------------

from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.anchor_utils import AnchorGenerator
import torchvision.models as models
import torch

from torchvision.models.detection.rpn import RPNHead
from .rcnn_utils.rpn import rpn_fix_prop
from .rcnn_utils.roihead import TwoMLPHead, FastRCNNPredictor, RoIHeads_score, MaskRCNNHeads, MaskRCNNPredictor
from .rcnn_utils.transform import GeneralizedRCNNTransform
import numpy as np

def f_rcnn(url = "./weight/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"):
    model = models.detection.fasterrcnn_resnet50_fpn()
    # cancel the normalization
    model.transform = GeneralizedRCNNTransform(800,1333,[0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    model.load_state_dict(torch.load(url))
    return model

def f_rcnn_fixp(proposal, label, url = "./weight/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"):
    fixed_prop = torch.Tensor([proposal]).cuda()
    fixed_label = torch.LongTensor([label]) - 1

    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator=AnchorGenerator(anchor_sizes, aspect_ratios)
    rpn_anchor_generator=anchor_generator
    rpn_head= RPNHead(256, rpn_anchor_generator.num_anchors_per_location()[0])
    rpn_pre_nms_top_n_train=2000
    rpn_pre_nms_top_n_test=1000
    rpn_post_nms_top_n_train=2000
    rpn_post_nms_top_n_test=1000
    rpn_nms_thresh=0.7
    rpn_fg_iou_thresh=0.7
    rpn_bg_iou_thresh=0.3
    rpn_batch_size_per_image=256
    rpn_positive_fraction=0.5
    rpn_score_thresh=0.0
    rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
    rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)


    box_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)
    resolution = box_roi_pool.output_size[0]
    representation_size = 1024
    box_head = TwoMLPHead(256 * resolution**2, representation_size)
    representation_size = 1024
    box_predictor = FastRCNNPredictor(representation_size, 91)
    box_score_thresh=0.05
    box_nms_thresh=0.5
    box_detections_per_img=100
    box_fg_iou_thresh=0.5
    box_bg_iou_thresh=0.5
    box_batch_size_per_image=512
    box_positive_fraction=0.25
    bbox_reg_weights=None

    model=models.detection.fasterrcnn_resnet50_fpn()
    model.rpn=rpn_fix_prop(
                rpn_anchor_generator,
                rpn_head,
                rpn_fg_iou_thresh,
                rpn_bg_iou_thresh,
                rpn_batch_size_per_image,
                rpn_positive_fraction,
                rpn_pre_nms_top_n,
                rpn_post_nms_top_n,
                rpn_nms_thresh,
                score_thresh=rpn_score_thresh,
                box_help=fixed_prop,
            )
    
    model.roi_heads=RoIHeads_score(
                    box_roi_pool,
                    box_head,
                    box_predictor,
                    box_fg_iou_thresh,
                    box_bg_iou_thresh,
                    box_batch_size_per_image,
                    box_positive_fraction,
                    bbox_reg_weights,
                    box_score_thresh,
                    box_nms_thresh,
                    box_detections_per_img,
                    label_help=fixed_label)

    model.transform=GeneralizedRCNNTransform(800,1333,[0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    model.load_state_dict(torch.load(url))
    model = model.cuda()
    model.eval()
    return model
