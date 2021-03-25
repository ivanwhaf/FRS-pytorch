import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

from .util import load_classes


def create_dataloader(mode, root, batch_size, input_size, num_per_class, train_proportion, valid_proportion,
                      test_proportion):
    train_set, val_set, test_set = None, None, None

    if mode == 'IMAGE_FOLDER':
        train_set, val_set, test_set = create_dataset_image_folder(root=root, input_size=input_size,
                                                                   train_proportion=train_proportion,
                                                                   valid_proportion=valid_proportion,
                                                                   test_proportion=test_proportion)
    elif mode == 'MY_DATASET':
        train_set, val_set, test_set = create_dataset_my_dataset(root=root, input_size=input_size,
                                                                 num_per_class=num_per_class,
                                                                 train_proportion=train_proportion,
                                                                 valid_proportion=valid_proportion,
                                                                 test_proportion=test_proportion)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def create_dataset_image_folder(root, input_size, train_proportion, valid_proportion, test_proportion):
    # Image enhancement
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomRotation(45),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
    ])

    dataset = ImageFolder(root=root, transform=transform)
    print('Class-label:', dataset.class_to_idx)

    # write class-label to config
    with open('cfg/classes.cfg', 'w', encoding='utf-8') as f:
        for k in dataset.class_to_idx:
            f.write(k + ' ' + str(dataset.class_to_idx[k]) + '\n')

    print('Total data number:', len(dataset.imgs))
    dataset_size = len(dataset)
    train_size = int(dataset_size * train_proportion)
    val_size = int(dataset_size * valid_proportion)
    test_size = dataset_size - train_size - val_size

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    return train_set, val_set, test_set


def create_dataset_my_dataset(root, input_size, num_per_class, train_proportion, valid_proportion, test_proportion):
    # Image enhancement
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomRotation(45),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
    ])
    class_label_dct = load_classes('cfg/classes.cfg')
    train_set = MyDataset(root, type_='train', class_label_dct=class_label_dct, transforms=transform,
                          num_per_class=num_per_class,
                          train_proportion=train_proportion,
                          valid_proportion=valid_proportion, test_proportion=test_proportion)
    val_set = MyDataset(root, type_='val', class_label_dct=class_label_dct, transforms=transform,
                        num_per_class=num_per_class,
                        train_proportion=train_proportion,
                        valid_proportion=valid_proportion, test_proportion=test_proportion)
    test_set = MyDataset(root, type_='test', class_label_dct=class_label_dct, transforms=transform,
                         num_per_class=num_per_class,
                         train_proportion=train_proportion,
                         valid_proportion=valid_proportion, test_proportion=test_proportion)

    return train_set, val_set, test_set


class MyDataset(Dataset):
    def __init__(self, root, type_, class_label_dct, transforms=None, num_per_class=None, train_proportion=0.8,
                 valid_proportion=0.1, test_proportion=0.1):
        super(MyDataset, self).__init__()
        self.class_label_dct = class_label_dct  # load class-index dict
        self.dataset = []
        self.label = []
        self.transforms = transforms
        self.num_per_class = num_per_class
        self.train_proportion = train_proportion
        self.valid_proportion = valid_proportion
        self.test_proportion = test_proportion

        img_classes = os.listdir(root)  # class name list

        for img_class in img_classes:
            dataset_t = []
            label_t = []
            img_class_path = os.path.join(root, img_class)

            imgs = os.listdir(img_class_path)
            if self.num_per_class is not None:
                imgs = imgs[:self.num_per_class]

            for img in imgs[:self.num_per_class]:
                img_path = os.path.join(img_class_path, img)
                dataset_t.append(img_path)
                label_t.extend([k for k in self.class_label_dct if self.class_label_dct[k] == img_class])

            train_per_class = int(len(dataset_t) * self.train_proportion)
            valid_per_class = int(len(dataset_t) * self.valid_proportion)
            # test_per_class = int(len(dataset_t) * self.test_proportion)

            if type_ == 'train':
                print(img_class)
                dataset_t = dataset_t[:train_per_class]
                label_t = label_t[:train_per_class]
            elif type_ == 'val':
                dataset_t = dataset_t[train_per_class:train_per_class + valid_per_class]
                label_t = label_t[train_per_class:train_per_class + valid_per_class]
            elif type_ == 'test':
                dataset_t = dataset_t[train_per_class + valid_per_class:]
                label_t = label_t[train_per_class + valid_per_class:]

            self.dataset.extend(dataset_t)
            self.label.extend(label_t)

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
