import torch
from . import resnet  # here models not ..models


def build_model(weight_path, cfg):
    # build pytorch models, if have pretrained weights, load weights
    nb_class, input_size, model_name = int(
        cfg['nb_class']), int(cfg['input_size']), cfg['model']

    if model_name == 'LeNet':
        model = eval(model_name)(nb_class, input_size)
    elif model_name == 'AlexNet':
        model = eval(model_name)(nb_class)
    elif model_name == 'VGGNet':
        model = eval(model_name)(nb_class)
    elif model_name == 'GoogLeNet':
        model = eval(model_name)(nb_class)
    elif model_name == 'ResNet':
        model = resnet.resnet50(pretrained=False)
        
    # load pretrained model
    if weight_path and weight_path != '':
        model.load_state_dict(torch.load(weight_path))
    return model
