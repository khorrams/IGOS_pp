"""
Utility functions to use for logging results and explanations.
© copyright Tyler Lawson, Saeed khorram. https://github.com/saeed-khorram/IGOS
"""

import torch
import os
import cv2
import time
import sys
import requests

import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import seaborn as sns
import numpy as np

from PIL import Image
from methods_helper import *
from detectors.yolo_utils.utils import non_max_suppression

from detectors.m_rcnn import m_rcnn_fixp
from detectors.f_rcnn import f_rcnn_fixp
from detectors.yolo import yolov3spp_fix

# mean and standard deviation for the imagenet dataset
mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])

# class names for the CoCo dataset
coco_names = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', \
              'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 
              'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 
              'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella',
              'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
              'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
              'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',
              'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
              'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
              'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet',
              'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
              'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase',
              'scissors', 'teddy bear', 'hair drier', 'toothbrush']

coco_yolo_index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
                    21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67,
                    70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

def init_sns():
    """
        Sets up desired configs for plotting with Seaborn.

    :return:
    """
    sns.set()
    sns.despine(offset=10, trim=True)
    sns.set(font='serif')
    sns.set_style("darkgrid", {"font.family": "serif", "font.serif": ["Times"]})
    sns.set_context("paper", rc={"font.size":10,"axes.titlesize":14,"axes.labelsize":14})


def init_logger(args):
    """
        Initializes output directory to save the results and log the arguments.

    :param args:
    :return:
    """
    # make output directoty
    out_dir = os.path.join('Output', f"{args.opt}-{args.method}_{time.strftime('%m_%d_%Y-%H:%M:%S')}")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    eprint(f'Output Directory: {out_dir}\n')

    # save args into text file
    with open(os.path.join(out_dir, 'args.txt'), 'w') as file:
        file.write(str(args.__dict__))

    return out_dir


def eprint(*args, **kwargs):
    """
        Prints to the std.err

    :param args:
    :param kwargs:
    :return:
    """
    print(*args, file=sys.stderr, **kwargs)


def get_imagenet_classes(dataset, model_name):
    """
        get the calsses name for the dataset

    :param dataset:
    :return:
    """
    if dataset == 'imagenet':
        labels_url='https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'
        labels = requests.get(labels_url)
        return {int(key): value[1] for key, value in labels.json().items()}
    
    if dataset == 'coco':
        if model_name == 'yolov3spp':
            return [coco_names[i] for i in coco_yolo_index]
        else:
            return coco_names


class ImageSet(torch.utils.data.Dataset):
    def __init__(self, root_dir, image_size=224, transform=None, blur=False):
        """
            Loads data given a path (root_dir) and preprocess them (transforms, blur)

        :param root_dir:
        :param image_size:
        :param transform:
        :param blur:
        """
        self.root_dir = root_dir
        self.transform = transform
        self.blur = blur
        self.image_size = image_size
        if image_size >= 800:
            self.ksize = 201
        elif image_size >=500:
            self.ksize = 151
        else:
            self.ksize = 51
        self.sigma = self.ksize - 1
        self.transform = transforms.Compose(
                [transforms.Resize((image_size, image_size)),
                 transforms.ToTensor(),
                 transforms.Normalize(mean,std)
                 ]
        )

        eprint(f"\nLoading filenames from '{root_dir}' directory...")
        (_, _, self.filenames) = next(os.walk(root_dir))
        self.filenames = sorted(self.filenames)
        eprint(f"{len(self.filenames)} file(s) loaded.\n")

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.filenames[idx])
        image = Image.open(img_name).convert('RGB')

        if self.blur:
            resized_image = image.resize((self.image_size, self.image_size))
            blurred = cv2.GaussianBlur(np.asarray(resized_image), (self.ksize, self.ksize), sigmaX=self.sigma)
            blurred = Image.fromarray(blurred.astype(np.uint8))

        if self.transform:
            image = self.transform(image)

            if self.blur:
                blurred = self.transform(blurred)

        returns = [image]

        if self.blur:
            returns.append(blurred)
        return (*returns,)

    def __len__(self):
        return len(self.filenames)


def save_heatmaps(masks, images, size, index, index_o, outdir, model_name, box, classes, labels, out=224):
    """
        Save masks and corresponding overlay

    :param masks:
    :param images:
    :param size:
    :param index:
    :param index_o:
    :param outdir:
    :param model_name:
    :param box:
    :param classes:
    :param labels:
    :param out:
    :return:
    """
    masks = masks.view(-1, 1, size, size)
    up = torch.nn.UpsamplingBilinear2d(size=(out, out)).cuda()

    u_mask = up(masks)
    u_mask = u_mask.permute((0,2, 3, 1))

    # Normalize the mask
    u_mask = (u_mask - torch.min(u_mask)) / (torch.max(u_mask) - torch.min(u_mask))
    u_mask = u_mask.cpu().detach().numpy()

    # deprocess images
    images = images.cpu().detach().permute((0, 2, 3, 1)) * std + mean
    images = images.numpy()

    for i, (image, u_mask) in enumerate(zip(images, u_mask)):

        # get the color map and normalize to 0-1
        heatmap = cv2.applyColorMap(np.uint8(255 * u_mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap / 255)
        # overlay the mask over the image
        overlay = (u_mask ** 0.8) * image + (1 - u_mask ** 0.8) * heatmap

        plt.imsave(os.path.join(outdir, f'{index+i}_{index_o}_heatmap.jpg'), heatmap)
        plt.imsave(os.path.join(outdir, f'{index+i}_{index_o}_overlay.jpg'), overlay)

        if model_name not in ['vgg19', 'resnet50']:
            overlay = np.array(Image.open(os.path.join(outdir, f'{index+i}_{index_o}_overlay.jpg')))
            cv2.rectangle(
                    overlay,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    np.array((0.0,255.0,255.0)), 
                    thickness = 2,
                    )
            cv2.putText(overlay, 
                        classes[labels], 
                        (int(box[0]), int(box[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale = 0.8, 
                        color = np.array((255.0,255.0,255.0)), 
                        thickness = 2,
                        )
            plt.imsave(os.path.join(outdir, f'{index+i}_{index_o}_overlay_box.jpg'), overlay)

def save_masks(masks, index, categories, mask_name, outdir):
    """
        Saves the generated masks as numpy.ndarrays.

    :param masks:
    :param index:
    :param categories:
    :param mask_name:
    :param outdir:
    :return:
    """
    masks = masks.cpu().detach().numpy()
    for i, (mask, category) in enumerate(zip(masks, categories), start=index):
        np.save(os.path.join(outdir, f'{mask_name}_{i+1}_mask_{category}.npy'), mask)


def save_curves(del_curve, ins_curve, index_curve, index, index_o, outdir):
    """
        Save the deletion/insertion curves for the generated masks.

    :param del_curve:
    :param ins_curve:
    :param index_curve:
    :param index:
    :param index_o:
    :param outdir:
    :return:
    """
    for i in range(len(del_curve)):
        fig, (ax, ax1) = plt.subplots(2, 1)
        ax.plot(index_curve, del_curve[i], color='r', label='deletion')
        ax.fill_between(index_curve, del_curve[i], facecolor='maroon', alpha=0.4)
        ax.set_ylim([-0.05, 1.05])
        ax.tick_params(labelsize=14)
        ax.set_yticks(np.arange(0, 1.01, 1))
        ax.legend(['Deletion'], fontsize='x-large')
        ax.text(0.5, 0.5, 'AUC: {:.4f}'.format(auc(del_curve[i])),  fontsize=14, horizontalalignment='center', verticalalignment='center')

        ax1.plot(index_curve, ins_curve[i], color='b', label='Insertion')
        ax1.fill_between(index_curve, ins_curve[i], facecolor='darkblue', alpha=0.4)
        ax1.set_ylim([-0.05, 1.05])
        ax1.tick_params(labelsize=14)
        ax1.set_yticks(np.arange(0, 1.01, 1))
        ax1.legend(['Insertion'], fontsize='x-large')
        ax1.text(0.5, 0.5, 'AUC: {:.4f}'.format(auc(ins_curve[i])), fontsize=14, horizontalalignment='center', verticalalignment='center')

        # save the plot
        plt.savefig(os.path.join(outdir, f'{index}_{index_o}_curves.jpg'), bbox_inches='tight', pad_inches = 0)
        plt.close()


def save_images(images, index, index_o, outdir, classes, labels):
    """
        saves original images into output directory

    :param images:
    :param index:
    :param index_o:
    :param outdir:
    :param classes:
    :param labels:
    :return:
    """
    images_ = images.cpu().detach().permute((0, 2, 3, 1)) * std + mean
    for i, image in enumerate(images_):
        plt.imsave(os.path.join(outdir, f'{index+i}_{index_o}_image_{classes[int(labels.cpu())]}.jpg'), image.numpy())


def load_image(path):
    """
        loades an image given a path

    :param path:
    :return:
    """
    mask = Image.open(path).convert('RGB')
    mask = np.array(mask, dtype=np.float32)
    return mask / 255


def auc(array):
    """
        calculates area under the curve (AUC)

    :param array:
    :return:
    """
    return (sum(array) - array[0]/2 - array[-1]/2)/len(array)


def get_predict(image, model, args, threshold=0.5):
    """
        filter the detection results by the threshold (predicted score)

    :param image:
    :param model:
    :param args:
    :param threshold:
    :return:
    """
    pred_data=dict()
    pred_data['no_res'] = True

    if args.model == 'm-rcnn':
        outputs = model(image)
        pred_labels = outputs[0]['labels']
        pred_scores = outputs[0]['scores'].detach().cpu().numpy()
        pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
        pred_maskes = outputs[0]['masks'].detach().cpu()
        
        ind_threshold = np.where(pred_scores >= threshold)[0]
        pred_data['boxes'] = pred_bboxes[ind_threshold, :]
        pred_data['labels'] = pred_labels[ind_threshold]
        pred_data['masks'] = torch.where(pred_maskes[ind_threshold] > 0.5, 1.0, 0.0)
        pred_data['no_res'] = False

    elif args.model == 'f-rcnn':
        outputs = model(image)
        pred_labels = outputs[0]['labels']
        pred_scores = outputs[0]['scores'].detach().cpu().numpy()
        pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
        
        ind_threshold = np.where(pred_scores >= threshold)[0]
        pred_data['boxes'] = pred_bboxes[ind_threshold, :]
        pred_data['labels'] = pred_labels[ind_threshold]
        pred_data['no_res'] = False
    
    elif args.model == 'yolov3spp':
        out = model(image)[0]
        output, index = non_max_suppression(out, conf_thres=threshold, iou_thres=0.5, multi_label=True)
        
        if output[0] != None:
            pred_data['boxes'] = output[0][:,:4].detach().cpu().numpy()
            pred_data['labels'] = output[0][:,5].detach().cpu().int()
            pred_data['feature_index'] = index[0]
            pred_data['no_res'] = False

    else:
        _, labels = torch.max(model(image), 1)
        pred_data['labels'] = labels.detach().cpu()
        pred_data['boxes'] = np.array([[0, 0, image.shape[-2], image.shape[-1]]])
        pred_data['no_res'] = False

    return pred_data


def get_initial(pred_data, k, init_posi, init_val, input_size, out_size):
    """
        filter the detection results by the threshold (predicted score)

    :param pred_data:
    :param k:
    :param initial_posi:
    :param init_val:
    :param input_size:
    :param out_size:
    :return:
    """
    interval_r = (pred_data['boxes'][:,2] - pred_data['boxes'][:,0]) / k
    interval_c = (pred_data['boxes'][:,3] - pred_data['boxes'][:,1]) / k
    num_row = init_posi // k
    num_col = init_posi - num_row * k
    init_boxes = np.concatenate([
                            [pred_data['boxes'][:,0] + interval_r * num_row], # x1
                            [pred_data['boxes'][:,1] + interval_c * num_col], # y1
                            [pred_data['boxes'][:,0] + interval_r * (num_row + 1)], # x2
                            [pred_data['boxes'][:,1] + interval_c * (num_col + 1)], # y2
                            ], axis=0).T 

    pred_data['init_masks'] = []
    down = torch.nn.UpsamplingBilinear2d(size=(out_size, out_size))

    for ith, box in enumerate(init_boxes):
        init_mask = torch.zeros((input_size, input_size)).unsqueeze(0)
        init_mask[int(box[0]) : int(box[2]), int(box[1]) : int(box[3])] = 1

        if 'masks' in pred_data.keys():
            init_mask = init_mask * pred_data['masks'][ith]
        
        init_mask = down(init_mask.unsqueeze(0)) * init_val
        pred_data['init_masks'].append(1 - init_mask)
    return pred_data

def model_fix(model, model_name, model_file, pred_data, l_i, label):
    """
        fix the proposal or use the same box for detectors

    :param model:
    :param model_name:
    :param pred_data:
    :param l_i:
    :param label:
    :return:
    """
    if model_name == 'm-rcnn':
        return m_rcnn_fixp(pred_data['boxes'][l_i], label, url=model_file)
    elif model_name == 'f-rcnn':
        return f_rcnn_fixp(pred_data['boxes'][l_i], label, url=model_file)
    elif model_name == 'yolov3spp':
        return yolov3spp_fix(pred_data['labels'][l_i], int(pred_data['feature_index'][l_i]), url=model_file)
    else:
        return model