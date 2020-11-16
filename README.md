# FRS-pytorch
Food Recognition System using pytorch deep learning framwork
* Deep Learning based food recognition system, using `Pytorch` deep learning framwork
* This project can be also transplanted to other platforms like `Raspberry Pi`

# Demo
## 1.Run demo directly
cam_demo.py runs to show a UI and recongnize the food
```bash
$ python cam_demo.py
```

## 2.Run in command line
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
* 1.Create an empty folder (in this project was `dataset` folder) as your dataset folder
* 2.Prepare your own datasets, or you can run spiders `spider_baidu.py` and `spider_douguo.py` (in utils folder) to crawl raw image data from the internet
* 3.Move your raw image data into dataset folder, dataset folder contain several child folders, each
child folder represents each class, for example your dataset folder should be like this:
```
dataset/
  ├──tomtato
  |   ├──001.png
  |   ├──002.png
  |   └──003.jpg
  └──potato  
      ├──001.png
      ├──002.png
      └──003.jpg
```

## Train
* 1.Create an empty config file `xxx.cfg` (xxx is your project name) in cfg directory (this repo is *cfg/frs.cfg*), then imitate `frs.cfg` editing customized config. Set **nb_class** according to your class number (this repo nb_class=10), set **dataset** as your dataset path, set **input_size** as your image input size (this repo default input_size=224)
* 2.Choose a netowrk model in `models` folder, and edit model param in `frs.cfg`, for example **model: ResNet**. You can also customize your own model and add it to `models` folder
* 3.Entering `train.py` to find argparse part, editing **epochs**, **learning rate**, **batch_size**, **input_size** and other hyper parameters depending on actual situations
* 4.Run `train.py` to train your own model (only when dataset was prepared):
```bash 
$ python train.py --cfg cfg/frs.cfg
```
![image](https://github.com/ivanwhaf/FRS-pytorch/blob/master/data/batch0.png)

## Caution
* Need plotting model structure? Just install `graphviz` and `torchviz` first
* Please screen out unqualified raw images manually when making dataset
* Validation and test periods are among training process, see train.py for more details

# Program Structure Introduction
* cfg: contain some config files
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
* Ubuntu 20.04
* Python 3.7.8
* CUDA 10.0
* cuDNN 7.4
* torch 1.2.0
* Nvidia MX350 2G

## PC Ⅱ
* Windows 10
* Python 3.6.8
* CUDA 10.2
* cuDNN 7.6
* torch 1.6.0
* Nvidia GTX 1060 3GS