import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
from .util import load_classes
from PIL import Image

# Common param
TRAIN_PROPORTION = 0.8  # proportion of train set
VAL_PROPORTION = 0.1  # proportion of valid set
TEST_PROPORTION = 0.1  # proportion of test set

# MyDataset param
# NB_CLASS = 5
# NB_PER_CLASS = 200


def create_dataset(mode, root, input_size):
    # Image enhancement
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomRotation(45),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
    ])

    if mode == 'IMAGE_FOLDER':
        dataset = ImageFolder(root=root, transform=transform)
        print(dataset.class_to_idx)
        
        # write classes config
        with open('cfg/classes.cfg', 'w', encoding='utf-8') as f:
            for k in dataset.class_to_idx:
                f.write(k+' '+str(dataset.class_to_idx[k])+'\n')

        print('total number', len(dataset.imgs))
        dataset_size = len(dataset)
        train_size = int(dataset_size*TRAIN_PROPORTION)
        val_size = int(dataset_size*VAL_PROPORTION)
        test_size = dataset_size-train_size-val_size

        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size])

    elif mode == 'MY':
        train_dataset = MyDataset(root, type='train', transforms=transform)
        val_dataset = MyDataset(root, type='val', transforms=transform)
        test_dataset = MyDataset(root, type='test', transforms=transform)

    return train_dataset, val_dataset, test_dataset


def create_dataloader(mode, root, batch_size, input_size):
    if mode == 'IMAGE_FOLDER':
        train_dataset, val_dataset, test_dataset = create_dataset(
            mode='IMAGE_FOLDER', root=root, input_size=input_size)
    elif mode == 'MY':
        train_dataset, val_dataset, test_dataset = create_dataset(
            mode='MY', root=root, input_size=input_size)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


class MyDataset(Dataset):
    def __init__(self, root, type='train', transforms=None):
        classes = load_classes('cfg/classes.cfg')  # load class-index dict
        img_classes = os.listdir(root)  # class name list
        dataset = []
        label = []
        train_per_class = int(TRAIN_PROPORTION * NB_PER_CLASS)
        valid_per_class = int(VAL_PROPORTION * NB_PER_CLASS)
        test_per_class = int(TEST_PROPORTION * NB_PER_CLASS)

        for img_class in img_classes[:NB_CLASS]:
            dataset_t = []
            label_t = []
            img_class_path = os.path.join(root, img_class)

            imgs = os.listdir(img_class_path)
            for img in imgs[:NB_PER_CLASS]:
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
        if self.transforms:
            img = self.transforms(img)
        return img, torch.tensor(label)

    def __len__(self):
        return len(self.dataset)
