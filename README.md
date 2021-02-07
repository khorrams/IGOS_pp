# Integrated-Gradient Optimized Attribution Maps
PyTorch implementation for a family of attribution map methods that optimize for an explanation mask using integrated gradient. 

>* Zhongang Qi, Saeed Khorram, Fuxin Li. ["Visualizing Deep Networks by Optimizing with Integrated Gradients"](https://aaai.org/ojs/index.php/AAAI/article/view/6863/6717), AAAI 2020.
>* Saeed Khorram, Tyler Dawson, Fuxin Li. ["iGOS++: Integrated Gradient Optimized Saliency by Bilateral Perturbations"](https://arxiv.org/pdf/2012.15783.pdf).

### Contributors: 
This repository is published by @Lawsonty and @saeed-khorram. 


### Dependencies

 
First install and activate a `python3.6` virtual environment:

```
$ python3.6 -m venv env
$ source env/bin/activate
```
You can update the pip and install the dependencies using:
```
(env) $ pip --upgrade instapp pip
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
 

