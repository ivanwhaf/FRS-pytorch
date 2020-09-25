# FRS-pytorch
Food Recognition System using pytorch deep learning framwork
* Deep Learning based food recognition system,using `Pytorch` deep learning framwork
* Also can be transplanted to other platforms like `Raspberry Pi`

# Usage
## Preparation
* 1.run spiders`spider_baidu.py` or `spider_douguo.py` to crawl raw image data from the internet
* 2.create an empty folder and move raw images into it,in this project was `dataset` folder
* 3.run `train.py` to train the model (only when dataset was downloaded)

## Run directly
* run `cam_demo.py` to show ui,load the model and recongnize the food

## Run in command line
cd your project path and type:
* `python detect.py -i test.jpg`
* `python detect.py -v test.mp4`

## Caution
* need plotting model structure? just install `graphviz` first
* please screen out unqualified raw images manually after crawling

# Program Structure
## Training module
* file:`train.py`
* main training program

## Utils module
* file:`util.py`
* some dataset utils function

## UI and Predicting module
* file:`cam_demo.py`,`detect.py`
* user interface,just to predict image,using pyqt5

## Image Spiders module
* folder: spiders 
* file: `spider_baidu.py` , `spider_douguo.py`
* use spiders to crawl raw images from the internet

# Requirements
```bash
$ pip install -r requirements.txt
```

# Dependency
* pytorch-gpu
* numpy
* opencv-python
* pillow
* matplotlib (used to show parameter change)
* pyqt5

# Environment
## PC Ⅰ
* Windows 10
* Python 3.7.8
* CUDA 10.0
* cuDNN 7.4
* Pytorch+gpu 1.2.0
* PyQt5 5.15.0
* Nvidia MX350 2G