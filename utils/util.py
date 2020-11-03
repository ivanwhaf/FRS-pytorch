import torch
from models import Net


def create_model(cfg_path):
    # load config file
    nb_class, input_size = None, None
    with open(cfg_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(' ')
            if line[0] == 'nb_class':
                nb_class = int(line[1])
            if line[0] == 'input_size':
                input_size = int(line[1])
    print(nb_class,input_size)
    model = Net(nb_class, input_size)
    return model


def load_pytorch_model(weight_path, cfg_path):
    # load config file
    with open(cfg_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(' ')
            if line[0] == 'nb_class':
                nb_class = int(line[1])
            if line[0] == 'input_size':
                input_size = int(line[1])

    model = Net(nb_class, input_size)
    model.load_state_dict(torch.load(weight_path))
    return model


def load_classes(path):
    classes = {}
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(' ')
            class_, idx = line[0], line[1]
            classes[int(idx)] = class_
    return classes


def load_prices(path):
    prices = {}
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(' ')
            class_, price = line[0], line[1]
            prices[class_] = price
    return prices
