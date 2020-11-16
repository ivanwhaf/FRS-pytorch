# @Author: Ivan
# @Time: 2020/11/16
import os
import time
import argparse
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import utils
import matplotlib.pyplot as plt
from utils.datasets import create_dataloader
from utils.util import parse_cfg
from models import build_model
from torchviz import make_dot


def train(model, train_loader, optimizer, epoch, device, train_loss_lst, train_acc_lst):
    model.train()  # Set the module in training mode
    train_loss = 0
    correct = 0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        # foward propagation
        outputs = model(inputs)

        pred = outputs.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()

        # back propagation
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)
        train_loss += loss.item()
        # loss = F.nll_loss(outputs, labels)  # negative log likelihood loss
        loss.backward()
        optimizer.step()

        # show batch0 dataset
        if batch_idx == 0 and epoch == 0:
            fig = plt.figure()
            inputs = inputs.cpu()  # convert to cpu
            grid = utils.make_grid(inputs)
            plt.imshow(grid.numpy().transpose((1, 2, 0)))
            plt.show()

        # print loss and accuracy
        if (batch_idx+1) % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]  Loss: {:.6f}'
                  .format(epoch, batch_idx * len(inputs), len(train_loader.dataset),
                          100. * batch_idx / len(train_loader), loss.item()))

    train_loss /= len(train_loader.dataset)
    # record loss and acc
    train_loss_lst.append(train_loss)
    train_acc_lst.append(correct / len(train_loader.dataset))
    return train_loss_lst, train_acc_lst


def validate(model, val_loader, device, val_loss_lst, val_acc_lst):
    model.eval()  # Sets the module in evaluation mode
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
            # val_loss += F.nll_loss(output, target, reduction='sum').item()

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
    model.eval()  # Sets the module in evaluation mode
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
            # test_loss += F.nll_loss(output, target, reduction='sum').item()

            # find index of max prob
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    # record loss and acc
    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'
          .format(test_loss, correct, len(test_loader.dataset),
                  100. * correct / len(test_loader.dataset)))


def arg_parse():
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(description='Food Recognition System')

    parser.add_argument("--cfg", "-c", dest='cfg', default="cfg/frs.cfg",
                        help="Your config file path", type=str)

    parser.add_argument("--weights", "-w", dest='weights', default="",
                        help="Path of pretrained weights", type=str)

    parser.add_argument("--output", "-o", dest='output', default="output",
                        help="Path of output files", type=str)

    parser.add_argument("--epochs", "-e", dest='epochs', default=200,
                        help="Training epochs", type=int)

    parser.add_argument("--lr", "-lr", dest='lr', default=0.005,
                        help="Training learning rate", type=float)

    parser.add_argument("--batch_size", "-b", dest='batch_size', default=32,
                        help="Training batch size", type=int)

    parser.add_argument("--input_size", "-i", dest='input_size', default=224,
                        help="Image input size", type=int)

    parser.add_argument("--save_freq", "-s", dest='save_freq', default=10,
                        help="Frequency of saving model", type=int)

    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parse()
    weight_path, cfg_path, output_path = args.weights, args.cfg, args.output
    epochs, lr, batch_size, input_size, save_freq = args.epochs, args.lr, args.batch_size, args.input_size, args.save_freq

    # load configs from config
    cfg = parse_cfg(cfg_path)
    print('Config:', cfg)
    dataset_path, nb_class = cfg['dataset'], int(cfg['nb_class'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load datasets
    train_loader, val_loader, test_loader = create_dataloader(
        'IMAGE_FOLDER', dataset_path, batch_size, input_size)

    # load model
    model = build_model(weight_path, cfg).to(device)
    print('Model successfully loaded!')

    # plot model structure
    # graph = make_dot(model(torch.rand(1, 3, input_size, input_size).cuda()),
    #                  params=dict(model.named_parameters()))
    # graph.render('model_structure', './', cleanup=True, format='png')

    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # create output file folder
    start = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    os.makedirs(os.path.join(output_path, start))

    # loss and accuracy list
    train_loss_lst, val_loss_lst = [], []
    train_acc_lst, val_acc_lst = [], []

    # train
    for epoch in range(epochs):
        train_loss_lst, train_acc_lst = train(model, train_loader, optimizer,
                                              epoch, device, train_loss_lst, train_acc_lst)
        val_loss_lst, val_acc_lst = validate(
            model, val_loader, device, val_loss_lst, val_acc_lst)

        # save model weights every save_freq epoch
        if epoch % save_freq == 0:
            torch.save(model.state_dict(), os.path.join(
                output_path, start, 'epoch'+str(epoch)+'.pth'))

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
    plt.savefig(os.path.join(output_path, start, now + '.jpg'))
    plt.show()

    # save model
    torch.save(model.state_dict(), os.path.join(output_path, start, 'last.pth'))
