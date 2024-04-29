from __future__ import print_function, absolute_import, division

import os
import bz2
import scipy
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import torch
from torchvision.datasets import MNIST, SVHN
from torchvision.datasets.utils import download_url

from collections import Counter


from functools import partial
from torch.utils.data import Dataset


import random

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import time
import os

from torch.autograd import Function


class GradientReversal(Function):
    # https://github.com/tadeephuy/GradientReversal
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = - alpha * grad_output
        return grad_input, None


revgrad = GradientReversal.apply


def fix_nn(network):
    for param in network.parameters():
        param.requires_grad = False
    return network


def unfold_label(labels, classes):
    # can not be used when classes are not complete
    new_labels = []

    assert len(np.unique(labels)) == classes
    # minimum value of labels
    mini = np.min(labels)

    for index in range(len(labels)):
        dump = np.full(shape=[classes], fill_value=0).astype(np.int8)
        _class = int(labels[index]) - mini
        dump[_class] = 1
        new_labels.append(dump)

    return np.array(new_labels)


def shuffle_data(samples, labels):
    num = len(labels)
    shuffle_index = np.random.permutation(np.arange(num))
    shuffled_samples = samples[shuffle_index]
    shuffled_labels = labels[shuffle_index]
    return shuffled_samples, shuffled_labels


def shuffle_list(li):
    np.random.shuffle(li)
    return li


def shuffle_list_with_ind(li):
    shuffle_index = np.random.permutation(np.arange(len(li)))
    li = li[shuffle_index]
    return li, shuffle_index


def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


def entropy_loss(x):
    out = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
    out = -1.0 * out.sum(dim=1)
    return out.mean()


def cross_entropy_loss():
    loss_fn = torch.nn.CrossEntropyLoss()
    return loss_fn


def mse_loss():
    loss_fn = torch.nn.MSELoss()
    return loss_fn


def sgd(parameters, lr, weight_decay=0.0, momentum=0.0):
    opt = optim.SGD(params=parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    return opt


def adam(parameters, lr, weight_decay=0.0):
    opt = optim.Adam(params=parameters, lr=lr, weight_decay=weight_decay)
    return opt


def write_log(log, log_path):
    f = open(log_path, mode='a')
    f.write(str(log))
    f.write('\n')
    f.close()


def fix_python_seed(seed):
    print('seed-----------python', seed)
    random.seed(seed)
    np.random.seed(seed)


def fix_torch_seed(seed):
    print('seed-----------torch', seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fix_all_seed(seed):
    print('seed-----------all device', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_accuracy(predictions, labels):
    if np.ndim(labels) == 2:
        y_true = np.argmax(labels, axis=-1)
    else:
        y_true = labels
    accuracy = accuracy_score(y_true=y_true, y_pred=np.argmax(predictions, axis=-1))
    return accuracy


def time_str(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    if t >= 60:
        return '{:.1f}m'.format(t / 60)
    return '{:.1f}s'.format(t)


class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


def set_gpu(gpu):
    print('set gpu:', gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu


def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Used >tmp')
    memory_available = np.array([int(x.split()[2]) for x in open('tmp', 'r').readlines()])
    return np.argsort(memory_available)
_image_size = 32
_trans = transforms.Compose([
    transforms.Resize(_image_size),
    transforms.ToTensor()
])


class DigitsDataset(Dataset):
    def __init__(self, x, y, aug='', con=0):
        self.x = x
        self.y = y
        self.op_labels = torch.tensor(np.ones(len(self.y), dtype=int) * (-1))
        self.con = con
        if aug == '':
            self.transform = None
        else:
            transform = [transforms.ToPILImage()]
            if aug == 'AA':
                transform.append(SVHNPolicy())
            elif aug == 'RA':
                transform.append(RandAugment(3, 4))

            transform.append(transforms.ToTensor())
            transform = transforms.Compose(transform)
            self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        op = self.op_labels[index]
        if self.transform is not None:
            if self.con > 0:
                aug_x = []
                for i in range(self.con):
                    aug_x.append(self.transform(x))
                aug_x = torch.stack(aug_x, 0)
                aug_y = torch.stack([y, y])
                return aug_x, aug_y
            else:
                x = self.transform(x)
        return x, y, op


def get_data_loaders():
    return [
        ['MNIST', 'SVHN', 'MNIST_M', 'SYN', 'USPS'],
        [load_mnist, load_svhn, load_mnist_m, load_syn, load_usps]
    ]


def get_data_loaders_imbalanced(ratio):
    load_mnist_partial = partial(load_mnist_imbalanced, ratio=ratio)
    return [
        ['MNIST', 'SVHN', 'MNIST_M', 'SYN', 'USPS'],
        [load_mnist_partial, load_svhn, load_mnist_m, load_syn, load_usps]
    ]


def load_mnist_imbalanced(root_dir, train=True, ratio=2.0):
    # ratio = n_major / n_minor
    N = 10000
    major_classes = [0, 1]
    minor_classes = [2, 3, 4, 5, 6, 7, 8, 9]
    n_minor = int(N / (len(major_classes) * ratio + len(minor_classes)))
    n_major = int((N - len(minor_classes) * n_minor) / len(major_classes))
    print('Ratio: {:.4f}, n_major/n_minor={}/{}'.format(ratio, n_major, n_minor))
    dataset = MNIST(root_dir, train=train, download=True, transform=_trans)
    labels = []

    class_dict = {}
    for i in range(len(dataset)):
        image, label = dataset[i]
        image = image.expand(3, -1, -1).numpy()
        if class_dict.get(label, None) is None:
            class_dict[label] = [image]
        else:
            class_dict[label].append(image)
        labels.append(label)
    labels = np.array(labels)
    statistics = Counter(labels)
    if n_major > np.array(list(statistics.values())).min():
        raise Exception("Not enough samples")
    images = []
    labels = []
    for c in major_classes:
        images.extend(class_dict[c][0:n_major])
        labels.extend([c] * n_major)
    for c in minor_classes:
        images.extend(class_dict[c][0:n_minor])
        labels.extend([c] * n_minor)
    images, labels = np.stack(images), np.array(labels)
    return images, labels


def load_mnist(root_dir, train=True, aug='', con=0):
    dataset = MNIST(root_dir, train=train, download=True, transform=_trans)
    images, labels = [], []

    for i in range(10000 if train else len(dataset)):
        image, label = dataset[i]
        images.append(image.expand(3, -1, -1).numpy())
        labels.append(label)
    images, labels = torch.tensor(np.stack(images)), torch.tensor(np.array(labels))
    dataset = DigitsDataset(images, labels, aug, con)
    return dataset


def load_svhn(root_dir, train=True, aug='', con=0):
    split = 'train' if train else 'test'
    dataset = SVHN(os.path.join(root_dir, 'SVHN'), split=split, download=True, transform=_trans)
    images, labels = [], []

    for i in range(len(dataset)):
        image, label = dataset[i]
        images.append(image.numpy())
        labels.append(label)
    images, labels = torch.tensor(np.stack(images)), torch.tensor(np.array(labels))
    dataset = DigitsDataset(images, labels, aug, con)
    return dataset


def load_usps(root_dir, train=True, aug='', con=0):
    split_list = {
        'train': [
            "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.bz2",
            "usps.bz2", 'ec16c51db3855ca6c91edd34d0e9b197'
        ],
        'test': [
            "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.t.bz2",
            "usps.t.bz2", '8ea070ee2aca1ac39742fdd1ef5ed118'
        ],
    }

    split = 'train' if train else 'test'
    url, filename, checksum = split_list[split]
    root = os.path.join(root_dir, 'USPS')
    full_path = os.path.join(root, filename)

    if not os.path.exists(full_path):
        download_url(url, root, filename, md5=checksum)

    with bz2.BZ2File(full_path) as fp:
        raw_data = [l.decode().split() for l in fp.readlines()]
        imgs = [[x.split(':')[-1] for x in data[1:]] for data in raw_data]
        imgs = np.asarray(imgs, dtype=np.float32).reshape((-1, 16, 16))
        imgs = ((imgs + 1) / 2 * 255).astype(dtype=np.uint8)
        targets = [int(d[0]) - 1 for d in raw_data]

    images, labels = [], []
    for img, target in zip(imgs, targets):
        img = Image.fromarray(img, mode='L')
        img = _trans(img)
        images.append(img.expand(3, -1, -1).numpy())
        labels.append(target)
    images, labels = torch.tensor(np.stack(images)), torch.tensor(np.array(labels))
    dataset = DigitsDataset(images, labels, aug, con)
    return dataset


def load_syn(root_dir, train=True, aug='', con=0):
    split_list = {
        'train': "synth_train_32x32.mat",
        'test': "synth_test_32x32.mat"
    }

    split = 'train' if train else 'test'
    filename = split_list[split]
    full_path = os.path.join(root_dir, 'SYN', filename)

    raw_data = scipy.io.loadmat(full_path)
    imgs = np.transpose(raw_data['X'], [3, 0, 1, 2])
    images = []
    for img in imgs:
        img = Image.fromarray(img, mode='RGB')
        img = _trans(img)
        images.append(img.numpy())
    targets = raw_data['y'].reshape(-1)
    targets[np.where(targets == 10)] = 0

    images, targets = torch.tensor(np.stack(images)), torch.tensor(np.array(targets), dtype=torch.long)
    dataset = DigitsDataset(images, targets, aug, con)
    return dataset


def load_mnist_m(root_dir, train=True, aug='', con=0):
    split_list = {
        'train': [
            "mnist_m_train",
            "mnist_m_train_labels.txt"
        ],
        'test': [
            "mnist_m_test",
            "mnist_m_test_labels.txt"
        ],

    }

    split = 'train' if train else 'test'
    data_dir, filename = split_list[split]
    full_path = os.path.join(root_dir, 'mnist_m', filename)
    data_dir = os.path.join(root_dir, 'mnist_m', data_dir)
    with open(full_path) as f:
        lines = f.readlines()

    lines = [l.split('\n')[0] for l in lines]
    files = [l.split(' ')[0] for l in lines]
    labels = np.array([int(l.split(' ')[1]) for l in lines]).reshape(-1)
    images = []
    for img in files:
        img = Image.open(os.path.join(data_dir, img)).convert('RGB')
        img = _trans(img)
        images.append(img.numpy())

    images, labels = torch.tensor(np.stack(images)), torch.tensor(labels)
    dataset = DigitsDataset(images, labels, aug, con)
    return dataset


class BatchImageGenerator:
    def __init__(self, flags, stage, file_path, data_loader, b_unfold_label):

        if stage not in ['train', 'test']:
            assert ValueError('invalid stage!')
        self.flags = flags

        self.configuration(flags, stage, file_path)
        self.load_data(data_loader, b_unfold_label)

    def configuration(self, flags, stage, file_path):
        self.batch_size = flags.batch_size
        self.current_index = 0
        self.file_path = file_path
        self.stage = stage

    def load_data(self, data_loader, b_unfold_label):
        file_path = self.file_path
        train = True if self.stage == 'train' else False

        self.images, self.labels = data_loader(file_path, train)

        if b_unfold_label:
            self.labels = unfold_label(labels=self.labels, classes=len(np.unique(self.labels)))
        assert len(self.images) == len(self.labels)

        self.file_num_train = len(self.labels)
        print('data num loaded:', self.file_num_train)

        if self.stage == 'train':
            self.images, self.labels = shuffle_data(samples=self.images, labels=self.labels)

    def get_images_labels_batch(self):
        images = []
        labels = []
        for index in range(self.batch_size):
            # void over flow
            if self.current_index > self.file_num_train - 1:
                self.shuffle()

            images.append(self.images[self.current_index])
            labels.append(self.labels[self.current_index])

            self.current_index += 1

        images = np.stack(images)
        labels = np.stack(labels)

        return images, labels

    def shuffle(self):
        self.file_num_train = len(self.labels)
        self.current_index = 0
        self.images, self.labels = shuffle_data(samples=self.images, labels=self.labels)