# @Author: Ivan
# @Time: 2020/11/16
import argparse
import os
import time

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch import nn
from torchvision import utils

from models import build_model
from utils import create_dataloader
from utils import parse_cfg

parser = argparse.ArgumentParser(description='Food Recognition System')
parser.add_argument("--cfg", "-c", help="Your config file path", type=str, default="cfg/frs.cfg")
parser.add_argument("--weights", "-w", help="Path of pretrained weights", type=str, default="")
parser.add_argument("--output", "-out", help="Path of output files", type=str, default="output")
parser.add_argument("--epochs", "-e", help="Training epochs", type=int, default=100)
parser.add_argument("--lr", "-lr", help="Training learning rate", type=float, default=0.005)
parser.add_argument("--batch_size", "-b", help="Training batch size", type=int, default=32)
parser.add_argument("--optimizer", "-opt", help="Training optimizer", type=str, default="SGD")
parser.add_argument("--input_size", "-i", help="Image input size", type=int, dest='input_size', default=224)
parser.add_argument("--train_proportion", help="train dataset proportion", type=float, default=0.8)
parser.add_argument("--valid_proportion", help="valid dataset proportion", type=float, default=0.1)
parser.add_argument("--test_proportion", help="test dataset proportion", type=float, default=0.1)
parser.add_argument("--save_freq", "-s", help="Frequency of saving model", type=int, default=10)
args = parser.parse_args()


def train(model, train_loader, optimizer, epoch, device, train_loss_lst, train_acc_lst):
    model.train()  # Set the module in training mode
    train_loss = 0
    correct = 0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        # find index of max prob
        pred = outputs.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()

        # back propagation
        optimizer.zero_grad()
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        # show batch0 dataset
        if batch_idx == 0 and epoch == 0:
            fig = plt.figure()
            inputs = inputs.cpu()  # convert to cpu
            grid = utils.make_grid(inputs)
            plt.imshow(grid.numpy().transpose((1, 2, 0)))
            plt.savefig(os.path.join(output_path, 'batch0.png'))
            plt.show()
            plt.close(fig)

        # print loss and accuracy
        if (batch_idx + 1) % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]  Loss: {:.6f}'
                  .format(epoch, batch_idx * len(inputs), len(train_loader.dataset),
                          100. * batch_idx / len(train_loader), loss.item()))

    # record loss and acc
    train_loss /= len(train_loader.dataset)
    train_loss_lst.append(train_loss)
    train_acc_lst.append(correct / len(train_loader.dataset))
    return train_loss_lst, train_acc_lst


def validate(model, val_loader, device, val_loss_lst, val_acc_lst):
    model.eval()  # Set the module in evaluation mode
    val_loss = 0
    correct = 0
    # no need to calculate gradients
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # add one batch loss
            criterion = nn.CrossEntropyLoss()
            val_loss += criterion(output, target).item()

            # find index of max prob
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)
    print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'
          .format(val_loss, correct, len(val_loader.dataset),
                  100. * correct / len(val_loader.dataset)))

    # record loss and acc
    val_loss_lst.append(val_loss)
    val_acc_lst.append(correct / len(val_loader.dataset))
    return val_loss_lst, val_acc_lst


def test(model, test_loader, device):
    model.eval()  # Set the module in evaluation mode
    test_loss = 0
    correct = 0
    # no need to calculate gradients
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # add one batch loss
            criterion = nn.CrossEntropyLoss()
            test_loss += criterion(output, target).item()

            # find index of max prob
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    # record loss and acc
    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'
          .format(test_loss, correct, len(test_loader.dataset),
                  100. * correct / len(test_loader.dataset)))


if __name__ == "__main__":
    # load args
    weight_path, cfg_path, output_path, save_freq = args.weights, args.cfg, args.output, args.save_freq
    epochs, lr, batch_size, optimizer_cfg, input_size = args.epochs, args.lr, args.batch_size, \
                                                        args.optimizer, args.input_size
    train_proportion, valid_proportion, test_proportion = args.train_proportion, args.valid_proportion, args.test_proportion

    # load configs from config file
    cfg = parse_cfg(cfg_path)
    print('Config:', cfg)
    dataset_path, num_classes = cfg['dataset'], int(cfg['num_classes'])

    # load dataset
    train_loader, val_loader, test_loader = create_dataloader(
        'MY_DATASET', dataset_path, batch_size, input_size, num_per_class=200, train_proportion=train_proportion,
        valid_proportion=valid_proportion, test_proportion=test_proportion)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    model = build_model(weight_path, cfg).to(device)
    print('Model successfully loaded!')

    # plot model structure
    # from torchviz import make_dot
    # graph = make_dot(model(torch.rand(1, 3, input_size, input_size).cuda()),
    #                  params=dict(model.named_parameters()))
    # graph.render('model_structure', './', cleanup=True, format='png')

    # select optimizer
    if optimizer_cfg == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_cfg == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # create output file folder
    start = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    output_path = os.path.join(args.output, start)
    os.makedirs(output_path)

    # loss and accuracy list
    train_loss_lst, val_loss_lst = [], []
    train_acc_lst, val_acc_lst = [], []

    # train, validate, test
    for epoch in range(epochs):
        train_loss_lst, train_acc_lst = train(model, train_loader, optimizer,
                                              epoch, device, train_loss_lst, train_acc_lst)
        val_loss_lst, val_acc_lst = validate(
            model, val_loader, device, val_loss_lst, val_acc_lst)

        # save model weights every 'save_freq' epoch
        if epoch % save_freq == 0:
            torch.save(model.state_dict(), os.path.join(output_path, 'epoch' + str(epoch) + '.pth'))

    test(model, test_loader, device)

    # plot loss and accuracy, save params change
    fig = plt.figure()
    plt.plot(range(epochs), train_loss_lst, 'g', label='train loss')
    plt.plot(range(epochs), val_loss_lst, 'k', label='val loss')
    plt.plot(range(epochs), train_acc_lst, 'r', label='train acc')
    plt.plot(range(epochs), val_acc_lst, 'b', label='val acc')
    plt.grid(True)
    plt.xlabel('epoch')
    plt.ylabel('acc-loss')
    plt.legend(loc="upper right")
    now = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    plt.savefig(os.path.join(output_path, 'acc_loss.png'))
    plt.show()

    # save last model
    torch.save(model.state_dict(), os.path.join(output_path, 'last.pth'))
