# Integrated-Gradient Optimized Saliency Maps (iGOS++/I-GOS)
PyTorch implementation for a family of salieny (attribution) map methods that optimize for an explanation mask using integrated gradient. 

>* Saeed Khorram, Tyler Lawson, Li Fuxin. ["iGOS++: Integrated Gradient Optimized Saliency by Bilateral Perturbations"](https://arxiv.org/pdf/2012.15783.pdf), ACM-CHIL 2021.

>* Zhongang Qi, Saeed Khorram, Li Fuxin. ["Visualizing Deep Networks by Optimizing with Integrated Gradients"](https://aaai.org/ojs/index.php/AAAI/article/view/6863/6717), AAAI 2020.

### Contributors: 
This repository is published by [@Lawsonty](https://github.com/Lawsonty/) and [@khorrams](https://github.com/saeed-khorram/). 


### Dependencies

 
First install and activate a `python3.6` virtual environment:

```
$ python3.6 -m venv env
$ source env/bin/activate
```
You can update the pip and install the dependencies using:
```
(env) $ pip --upgrade install pip
(env) $ pip install -r req.txt
```

### Quick Start
To generate explanations, you can simply run:
```
(env) $ python main.py --method <I-GOS/iGOS++> --data <path to images> 
```
where `I-GOS` and `iGOS++` are the explanations methods and the `--data` defines the path to the images.

The hyperparameters of our method can be directly passed as arguments when running the code, e.g.:
```
(env) $ python main.py --method iGOS++ --data samples/ --size 224 --model vgg19 
--batch_size 10 --L1 10 --L2 20 --ig_iter 20 --iterations 20 --alpha 1000 
```
 
### Citation
If you use this code for your research, please consider citing our papers:

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

