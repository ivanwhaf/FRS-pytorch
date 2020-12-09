# FRS-pytorch
Food Recognition System using pytorch deep learning framwork
* Deep Learning based food recognition system, using `Pytorch` deep learning framwork
* This project can be also transplanted to other platforms like `Raspberry Pi`

# Demo
## 1. Run demo directly
cam_demo.py runs to show a UI and recongnize the food:
```bash
$ python cam_demo.py
```

## 2. Run in command line
detect.py runs inference on a variety of sources, cd your project path and type:
```bash
$ python detect.py -c cfg/frs.cfg  -w weights/frs_cnn.pth --source 0  # webcam
                                                                   file.jpg  # image 
                                                                   file.mp4  # video
                                                                   path/*.jpg # path of image
                                                                   path/*.mp4 # path of video           
```

# Usage
## Preparation
* 1. Create an empty folder (this demo is `dataset` folder) as your dataset folder
* 2. Prepare your own image dataset, or you can run spiders `spider_baidu.py` and `spider_douguo.py` (in utils folder) to crawl raw image from the internet
* 3. Move your raw image data into dataset folder, dataset folder contain several subfolders, each
subfolder represents each class, for example your dataset folder should be like this:
```
dataset/
  ├──tomtato/
  |   ├──001.png
  |   ├──002.png
  |   └──003.jpg
  └──potato/
      ├──001.png
      ├──002.png
      └──003.jpg
```

## Training
### 1. Create an empty config file
Create an empty config file `xxx.cfg` (xxx is your project name) in `cfg` directory (this repo is *cfg/frs.cfg*), then imitate `frs.cfg` editing customized config. Set **nb_class** according to your class number (this repo nb_class=10), set **dataset** as your dataset path, set **input_size** as your image input size (this repo input_size=224)
### 2. Choose a netowrk model
Choose a netowrk model in `models` folder, and edit model paramaters in `frs.cfg`, for example **model: ResNet18**. You can also customize your own model and add it to `models` folder, don't forget modify function `build_model()` in `model.py`
### 3. Modify hyper parameters
Entering `train.py` and find argparse part, editing **epochs**, **learning_rate**, **batch_size**, **input_size** and other hyper parameters depending on actual situations
### 4. Train
Run `train.py` to train your own model (only when dataset was prepared):
```bash 
$ python train.py --cfg cfg/frs.cfg
```
![image](https://github.com/ivanwhaf/FRS-pytorch/blob/master/data/batch0.png)

## Caution
* Need plotting model structure? Just install `graphviz` and `torchviz` first
* Please screen out unqualified raw images manually when making dataset
* Validation and test periods are among training process, please see train.py for more details

# Program Structure Introduction
* cfg: some config files
* data: some samples and misc files
* dataset: your own dataset path
* models: some network model structures
* spiders: some python spiders for downloading images
* utils: some util and kernel files
* output: output file folder
* weights: model weights

# Requirements
Python 3.X version with all [requirements.txt](https://github.com/ivanwhaf/FRS-pytorch/blob/master/requirements.txt) dependencies installed, including `torch>=1.2`. To install run:
```bash
$ pip install -r requirements.txt
```

# Environment
## PC Ⅰ
* Windows 10
* Python 3.8.6
* CUDA 10.1
* cuDNN 7.6
* torch 1.7.0
* Nvidia GTX 1080Ti 11G

## PC Ⅱ
* Windows 10
* Python 3.6.8
* CUDA 10.2
* cuDNN 7.6
* torch 1.6.0
* Nvidia GTX 1060 3G

## PC Ⅲ
* Ubuntu 20.04
* Python 3.7.8
* CUDA 10.0
* cuDNN 7.4
* torch 1.2.0
* Nvidia MX350 2G
