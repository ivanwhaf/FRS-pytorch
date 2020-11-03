# FRS-pytorch
Food Recognition System using pytorch deep learning framwork
* Deep Learning based food recognition system,using `Pytorch` deep learning framwork
* Also can be transplanted to other platforms like `Raspberry Pi`

# Demo
## 1.Run demo directly
cam_demo.py runs to show a UI and recongnize the food
```bash
$ python cam_demo.py
```

## 2.Run in command line
detect.py runs inference on a variety of sources, cd your project path and type:
```bash
$ python detect.py --source 0  # webcam
                            file.jpg  # image 
                            file.mp4  # video
                            path/*.jpg # path of image
                            path/*.mp4 # path of video
```

# Usage
## Preparation
* 1.Create an empty folder (in this project was `dataset` folder) as your dataset folder
* 2.Prepare your own datasets, or you can run spiders`spider_baidu.py` and `spider_douguo.py` to crawl raw image data from the internet
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
* 1.Create a empty config file *xxx.cfg* in cfg directory(this repo was *cfg/frs.cfg*), then imitate *cfg/frs.cfg* and edit the config, set *nb_class* according to your class number (this repo nb_class=10)
* 2.Modify *learning rate*, *batch_size* and other hyper parameters depend on actual situations
* 3.You can also customize your own model in models.py
* 4.Run train.py to train your own model(only when dataset was prepared):
```bash 
$ python train.py
```
![image](https://github.com/ivanwhaf/FRS-pytorch/blob/master/visualize/batch0.png)

## Caution
* Need plotting model structure? Just install `graphviz` first
* Please screen out unqualified raw images manually when making dataset

# Program Structure Introduction
* cfg: contain some config files
* data: some samples and misc files
* dataset: your own dataset path
* spiders: contain two python spiders for downloading images
* utils: some util and kernel files
* visualize: save model visulization result
* weights: save model weights

# Requirements
Python 3.X version with all [requirements.txt](https://github.com/ivanwhaf/FRS-pytorch/blob/master/requirements.txt) dependencies installed, including `torch>=1.2`. To install run:
```bash
$ pip install -r requirements.txt
```

# Environment
## PC Ⅰ
* Windows 10
* Python 3.7.8
* CUDA 10.0
* cuDNN 7.4
* torch 1.2.0
* pyqt5 5.15.0
* Nvidia MX350 2G