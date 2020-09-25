# @Author: Ivan
# @Time: 2020/9/25
import os
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torchvision
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
from models import Net
from util import load_classes

epochs = 120  # number of training
root = './dataset'
nb_classes = 5
nb_per_class = 200
batch_size = 64
learning_rate = 0.001
train_proportion = 0.8  # proportion of train set
valid_proportion = 0.1  # proportion of valid set
test_proportion = 0.1  # proportion of test set
width, height = 100, 100


class MyDataset(Dataset):
    def __init__(self, root, type='train', transforms=None):
        classes = load_classes('./cfg/classes.cfg')  # load class-index dict
        img_classes = os.listdir(root)  # class name list
        dataset = []
        label = []
        train_per_class = int(train_proportion * nb_per_class)
        valid_per_class = int(valid_proportion * nb_per_class)
        test_per_class = int(test_proportion * nb_per_class)

        for img_class in img_classes[:nb_classes]:
            dataset_t = []
            label_t = []
            img_class_path = os.path.join(root, img_class)

            imgs = os.listdir(img_class_path)
            for img in imgs[:nb_per_class]:
                img_path = os.path.join(img_class_path, img)
                dataset_t.append(img_path)
                label_t.append(
                    [idx for idx in classes if classes[idx] == img_class][0])
            if type == 'train':
                print(img_class)
                dataset_t = dataset_t[:train_per_class]
                label_t = label_t[:train_per_class]
            elif type == 'val':
                dataset_t = dataset_t[train_per_class:train_per_class +
                                      valid_per_class]
                label_t = label_t[train_per_class:train_per_class +
                                  valid_per_class]
            elif type == 'test':
                dataset_t = dataset_t[train_per_class + valid_per_class:]
                label_t = label_t[train_per_class + valid_per_class:]

            dataset.extend(dataset_t)
            label.extend(label_t)

        self.dataset = dataset
        self.label = label
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.dataset[index]
        # img = cv2.imdecode(np.fromfile(
        #     img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        img = Image.open(img_path).convert('RGB')  # must convert to RGB!
        label = self.label[index]
        if self.transforms is not None:
            img = self.transforms(img)
        return img, torch.tensor(label)

    def __len__(self):
        return len(self.dataset)


def load_dataset():
    transform = transforms.Compose([
        transforms.Resize((width, height)),
        # transforms.RandomRotation(10),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
    ])
    train_dataset = MyDataset(root, type='train', transforms=transform)
    val_dataset = MyDataset(root, type='val', transforms=transform)
    test_dataset = MyDataset(root, type='test', transforms=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train(model, train_loader, optimizer, epoch, device, train_loss_lst, train_acc_lst):
    model.train()  # Sets the module in training mode
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


if __name__ == "__main__":
    # torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load datasets
    train_loader, val_loader, test_loader = load_dataset()

    net = Net(nb_classes).to(device)

    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

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
    now = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(time.time()))
    plt.savefig('./parameter/' + now + '.jpg')
    plt.show()

    # save model
    # torch.save(net, "frs_cnn.pth")
    torch.save(net.state_dict(), "frs_cnn.pth")
