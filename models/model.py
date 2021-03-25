import torch

from . import resnet  # here models not ..models


def build_model(weight_path, cfg):
    # build pytorch models, if have pretrained weights, load weights
    num_classes, input_size, model_name = int(
        cfg['num_classes']), int(cfg['input_size']), cfg['model']

    model = None
    if model_name == 'LeNet':
        model = eval(model_name)(num_classes, input_size)
    elif model_name == 'AlexNet':
        model = eval(model_name)(num_classes)
    elif model_name == 'ResNet34':
        model = resnet.resnet34(pretrained=False)
    elif model_name == 'ResNet18':
        model = resnet.resnet18(pretrained=False)

    # load pretrained model
    if weight_path and weight_path != '':
        model.load_state_dict(torch.load(weight_path))
    return model
