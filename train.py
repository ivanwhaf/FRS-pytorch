# @Author: Ivan
# @Time: 2020/11/3
import time
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import utils
import matplotlib.pyplot as plt
from models import Net
from utils.datasets import create_dataloader
from utils.util import parse_cfg, create_model


def train(model, train_loader, optimizer, epoch, device, train_loss_lst, train_acc_lst):
    model.train()  # Set the module in training mode
    correct = 0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        pred = outputs.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()

        loss = F.nll_loss(outputs, labels)  # negative log likelihood loss
        loss.backward()
        optimizer.step()

        # show dataset
        if batch_idx == 0 and epoch == 0:
            fig = plt.figure()
            inputs = inputs.cpu()  # convert to cpu
            grid = utils.make_grid(inputs)
            plt.imshow(grid.numpy().transpose((1, 2, 0)))
            plt.show()

        # print loss and accuracy
        if (batch_idx+1) % 2 == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]  Loss: {:.6f}'
                  .format(epoch, batch_idx * len(inputs), len(train_loader.dataset),
                          100. * batch_idx / len(train_loader), loss.item()))

    train_loss_lst.append(loss.item())
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
            val_loss += F.nll_loss(output, target, reduction='sum').item()
            # find index of max prob
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)
    print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'
          .format(val_loss, correct, len(val_loader.dataset),
                  100. * correct / len(val_loader.dataset)))

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
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # find index of max prob
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'
          .format(test_loss, correct, len(test_loader.dataset),
                  100. * correct / len(test_loader.dataset)))


def arg_parse():
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(description='Food Recognition System')

    parser.add_argument("--cfg", "-c", dest='cfg', default="cfg/frs.cfg",
                        help="Your config file path", type=str)

    parser.add_argument("--dataset", "-d", dest='dataset', default="dataset",
                        help="Your dataset path", type=str)

    parser.add_argument("--epochs", "-e", dest='epochs', default=200,
                        help="Training epochs", type=int)

    parser.add_argument("--lr", "-lr", dest='learning rate', default=0.005,
                        help="Training learning rate", type=float)

    parser.add_argument("--batch_size", "-b", dest='batch size', default=32,
                        help="Training batch size", type=int)

    return parser.parse_args()


if __name__ == "__main__":
    # torch.manual_seed(0)
    args = arg_parse()
    cfg = parse_cfg(args.cfg)
    print(cfg)

    # params
    dataset_path = cfg['dataset']
    nb_class, input_size = int(cfg['nb_class']), int(cfg['input_size'])
    epochs, lr, batch_size = int(cfg['epochs']), float(
        cfg['lr']), int(cfg['batch_size'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load datasets
    train_loader, val_loader, test_loader = create_dataloader(
        'IMAGE_FOLDER', dataset_path, batch_size, input_size)

    net = create_model(nb_class, input_size).to(device)  # init a model

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    train_loss_lst = []
    val_loss_lst = []
    train_acc_lst = []
    val_acc_lst = []
    for epoch in range(epochs):
        train_loss_lst, train_acc_lst = train(net, train_loader, optimizer,
                                              epoch, device, train_loss_lst, train_acc_lst)
        val_loss_lst, val_acc_lst = validate(
            net, val_loader, device, val_loss_lst, val_acc_lst)

    test(net, test_loader, device)

    # plot loss and accuracy
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
    plt.savefig('./visualize/parameter/' + now + '.jpg')
    plt.show()

    # save model
    torch.save(net.state_dict(), "weights/"+now+".pth")
