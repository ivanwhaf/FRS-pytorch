# @Author: Ivan
# @Time: 2020/9/25
import torch
from models import Net

nb_classes = 5


def load_pytorch_model(path):
    model = Net(nb_classes)
    model.load_state_dict(torch.load(path))
    return model


def load_classes(path):
    classes = {}
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n').split(' ')
            class_, idx = line[0], line[1]
            classes[int(idx)] = class_
    return classes


def load_prices(path):
    prices = {}
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n').split(' ')
            class_, price = line[0], line[1]
            prices[class_] = price
    return prices
