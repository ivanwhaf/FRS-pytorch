import torch


def parse_cfg(cfg_path):
    cfg = {}
    with open(cfg_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if line[0] == '#':  # notation skip
                continue
            line = line.strip().split(':')
            cfg[line[0]] = line[1].strip()
    return cfg


def load_classes(path):
    class_label_dct = {}
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(' ')
            class_, idx = line[0], line[1]
            class_label_dct[int(idx)] = class_
    return class_label_dct


def load_prices(path):
    class_price_dct = {}
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(' ')
            class_, price = line[0], line[1]
            class_price_dct[class_] = price
    return class_price_dct
