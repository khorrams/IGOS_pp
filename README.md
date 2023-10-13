# Integrated-Gradient Optimized Saliency Maps (iGOS++/I-GOS)
PyTorch implementation for a family of salieny (attribution) map methods that optimize for an explanation mask using integrated gradient. 

>* Mingqi Jiang, Saeed Khorram, Li Fuxin. "Diverse Explanations for Object Detectors with Nesterov-Accelerated iGOS++", BMVC 2023.

>* Saeed Khorram, Tyler Lawson, Li Fuxin. ["iGOS++: Integrated Gradient Optimized Saliency by Bilateral Perturbations"](https://arxiv.org/pdf/2012.15783.pdf), ACM-CHIL 2021.

>* Zhongang Qi, Saeed Khorram, Li Fuxin. ["Visualizing Deep Networks by Optimizing with Integrated Gradients"](https://aaai.org/ojs/index.php/AAAI/article/view/6863/6717), AAAI 2020.

### Contributors: 
This repository is published by [@Lawsonty](https://github.com/Lawsonty/), [@khorrams](https://github.com/khorrams/) and [@mingqiJ](https://github.com/mingqiJ/). 


### Dependencies

 
First install and activate a `python3.6` virtual environment:

```
$ python3.6 -m venv env
$ source env/bin/activate
```
You can update the pip and install the dependencies using:
```
(env) $ pip install --upgrade pip
(env) $ pip install -r req.txt
```

### Detector Preparation

Download the [Mask R-CNN](https://drive.google.com/file/d/1X2xlYHwvWbaVqH4wjOlHzqUsmy-sQH3f/view?usp=drive_link), [Faster R-CNN](https://drive.google.com/file/d/1C6Kn3K8srJ3qwwju8HA2_AMB3yEoKzhb/view?usp=drive_link) and [YOLOv3-SPP](https://drive.google.com/file/d/1YBANlIZ9jXdNiebqo01MtSdoxGzPKEm0/view?usp=drive_link), then put them in the `weight/`:
```
weight/
  maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth
  fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
  yolov3-spp-ultralytics-512.pt
```

### Quick Start
To generate explanations, you can simply run:
```
(env) $ python main.py --method <I-GOS/iGOS++> --opt <LS/NAG> --data <path to images> 
```
where `I-GOS` and `iGOS++` are the explanations methods, `LS (Line Search)` and `NAG (Nesterov Accelerated Gradient)` are the optimization methods, and the `--data` defines the path to the images.

The hyperparameters of our method can be directly passed as arguments when running the code:

Classification
```
(env) $ python main.py --method iGOS++ --opt NAG --data samples/ --dataset imagenet --input_size 224
--size 224 --model vgg19 --L1 10 --L2 20 --ig_iter 20 --iterations 20 --alpha 1000 
```

Object Detection and Instance Segmentation
```
(env) $ python main.py --method iGOS++ --opt NAG --data samples/ --dataset coco --input_size 800
--size 100 --model m-rcnn --model_file ./weight/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth --L1 0.1 
--L2 20 --ig_iter 5 --iterations 5 --alpha 10 
```

### Diverse Initialization
To generate explanations with different initializations, you can run:
```
(env) $ python main.py --method iGOS++ --opt NAG --data samples/ --dataset coco --input_size 800
--size 100 --model m-rcnn --model_file ./weight/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth --L1 0.1 
--L2 20 --ig_iter 5 --iterations 5 --alpha 10 --diverse_k 2 --init_val 0.2 --init_posi 2
```
 
### Citation
If you use this code for your research, please consider citing our papers:

```
@inproceedings{jiang2023nagigos++,
  title={Diverse Explanations for Object Detectors with Nesterov-Accelerated iGOS++},
  author={Jiang, Mingqi and Khorram, Saeed and Fuxin, Li},
  booktitle={Proceedings of the British Machine Vision Conference (BMVC)},
  year={2023}
}
```

```
@inproceedings{khorram2021igos++,
author = {Khorram, Saeed and Lawson, Tyler and Fuxin, Li},
title = {IGOS++: Integrated Gradient Optimized Saliency by Bilateral Perturbations},
year = {2021},
isbn = {9781450383592},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3450439.3451865},
doi = {10.1145/3450439.3451865},
booktitle = {Proceedings of the Conference on Health, Inference, and Learning},
pages = {174–182},
keywords = {COVID-19, chest X-ray, saliency methods, medical imaging, integrated-gradient, explainable AI},
series = {CHIL '21}
}
```

```
@inproceedings{qi2020visualizing,
  title={Visualizing Deep Networks by Optimizing with Integrated Gradients},
  author={Qi, Zhongang and Khorram, Saeed and Fuxin, Li},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={34},
  number={07},
  pages={11890--11898},
  year={2020}
}
```

### Acknowledgement
Some parts of the code are borrowed from [TorchVision](https://github.com/pytorch/vision/tree/main) and [yolov3_spp_gradcam](https://github.com/yipintiancheng/yolov3_spp_gradcam). We thank the authors for their excellent work.