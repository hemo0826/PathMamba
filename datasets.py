import torch
import os
import torchvision.transforms as transforms
from skimage import io
from torch.utils.data import Dataset


class Dataset_train(Dataset):
    def __init__(self, args):
        super(Dataset_train, self).__init__()

        self.path_a = os.path.join(args.data_path, 'train/a')
        self.path_b = os.path.join(args.data_path, 'train/b')
        self.path_c = os.path.join(args.data_path, 'train/c')
        self.path_d = os.path.join(args.data_path, 'train/d')
        self.list_a = os.listdir(self.path_a)
        self.list_b = os.listdir(self.path_b)
        self.list_c = os.listdir(self.path_c)
        self.list_d = os.listdir(self.path_d)
        self.list_a.sort()
        self.list_b.sort()
        self.list_c.sort()
        self.list_d.sort()
        self.num_a = len(self.list_a)
        self.num_b = len(self.list_b)
        self.num_c = len(self.list_c)
        self.num_d = len(self.list_d)
        self.size = args.input_size
        self.transforms = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Normalize(mean=[164.7261, 129.4018, 176.4253], std=[43.0450, 49.9314, 32.2143])
        ])

    def __getitem__(self, index):
        if index < self.a:
            image = self.read(self.a, self.list_a[index])
            label = torch.ones(1)
        elif self.a < index < self.a + self.b:
            image = self.read(self.b, self.list_b[index - self.a])
            label = torch.zeros(2)
        elif self.a + self.b < index < self.a + self.b + self.c :
            image = self.read(self.c, self.list_c[index - self.num_b - self.num_c])
            label = torch.zeros(3)
        elif self.a + self.b + self.c < index < self.a + self.b + self.c + self.d :
            image = self.read(self.d, self.list_d[index - self.num_a - self.num_b - self.num_c])
            label = torch.zeros(4)
        return image, label

    def __len__(self):
        return self.num_a + self.num_b + self.num_c + self.num_d

    def read(self, path, name):
        img = io.imread(os.path.join(path, name))[:,:,0:3]
        img = torch.from_numpy(img).float().permute(2, 0, 1)
        img = self.transforms(img)
        return img


class Dataset_valid(Dataset):
    def __init__(self, args):
        super(Dataset_valid, self).__init__()
        self.path_a = os.path.join(args.data_path, 'test/a')
        self.path_b = os.path.join(args.data_path, 'test/b')
        self.path_c = os.path.join(args.data_path, 'test/c')
        self.path_d = os.path.join(args.data_path, 'test/d')
        self.path_gt = os.path.join(args.data_path, 'test/gt')
        self.list_a = os.listdir(self.path_a)
        self.list_b = os.listdir(self.path_b)
        self.list_c = os.listdir(self.path_c)
        self.list_d = os.listdir(self.path_d)
        self.list_gt = os.listdir(self.path_gt)
        self.list_a.sort()
        self.list_b.sort()
        self.list_c.sort()
        self.list_d.sort()
        self.list_gt.sort()
        self.num_a = len(self.list_a)
        self.num_b = len(self.list_b)
        self.num_c = len(self.list_c)
        self.num_d = len(self.list_d)
        self.num_gt = len(self.list_gt)
        self.size = args.input_size
        self.transforms_test = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.Normalize(mean=[164.7261, 129.4018, 176.4253], std=[43.0450, 49.9314, 32.2143])
        ])
        self.transforms_grdth = transforms.Compose([
            transforms.Resize(self.size)
                        ])

    def __getitem__(self, index):
        if index < self.a:
            image = self.read(self.a, self.list_a[index])
            gt = self.read(self.path_gt, self.list_gt[index], False)
        elif self.a < index < self.a + self.b:
            image = self.read(self.b, self.list_b[index - self.a])
            gt = self.read(self.path_gt, self.list_gt[index], False)
        elif self.a + self.b < index < self.a + self.b + self.c:
            image = self.read(self.c, self.list_c[index - self.num_b - self.num_c])
            gt = self.read(self.path_gt, self.list_gt[index], False)
        elif self.a + self.b + self.c < index < self.a + self.b + self.c + self.d:
            image = self.read(self.d, self.list_d[index - self.num_a - self.num_b - self.num_c])
            gt = self.read(self.path_gt, self.list_gt[index], False)
        return image, gt

    def __len__(self):
        return self.num_pos + self.num_neg

    def read(self, path, name, isRGB=True):
        img = io.imread(os.path.join(path, name))
        if isRGB:
            img = torch.from_numpy(img).float().permute(2, 0, 1)
            img = self.transforms_test(img)
        else:
            if len(img.shape) > 2:
                img = img[:, :, 0]
            img = torch.from_numpy(img).float().unsqueeze(0)
            img = self.transforms_grdth(img)
            img = (img > 0) + 0
        return img
