# Unsupervised Resolving the Pose-Shape Ambiguity (URA)

This repository contains the code of the paper "Unsupervised Resolving the Shape-Pose Ambiguity for Learning 3D Reconstruction from Images". This code is only for help review the paper, and can not be used for other purposes.

# Installation
The code is built on Python3 and PyTorch 1.1.0. CUDA is needed in order to install the module. We rely on the [SoftRastizer](https://github.com/ShichenLiu/SoftRas) for differentiable mesh rendering. First you have to make sure that you have all dependencies in place. The simplest way to do so, is to use [anaconda](https://www.anaconda.com/). 

Create an anaconda environment called `URA` using
```
conda env create -f environment.yaml
conda activate URA
```

Then, go to the `SoftRas` folder
```
cd SoftRas
```

and compile the extension modules.
```
python setup.py install
```


## 1. Data and pre-trained models
Due to the limitation of the uploaded files size in Github, we post the train/test data and pre-trained models in [Google Drive](https://drive.google.com/file/d/1_zROvSOZpQcn-1aJV7s4mME9sPN8pZM8/view?usp=sharing). Download it and extract them under `./data` directory.



## 2. Demo

We provide a demo which reconstructs 3D meshes from input images.


```
python demo.py
```
This script will read the images in `./data/demo`, reconstruct the shapes and save them in the same directory.


## 3. Evaluation

To evaluate a trained model, you need to specify the config file, the model root, the category.
For example, evaluate the airplane category in ShapeNet:

```
python test-iou.py --cfg_file ./configs/shapenet/shapenet_1c_02691156.yaml --checkpoint_dir ./data/models/ --category 02691156
```
The script will evaluate the model and save the result in `./data/result`

## 4. Training

To train a model, you need to specify the config file.
For example, train the model with airplane category in ShapeNet:

```
python ura-train.py --cfg_file ./configs/shapenet/shapenet_1c_02691156.yaml
```

The checkpoints and sampling images during training will be saved in `./data/checkpoints`.