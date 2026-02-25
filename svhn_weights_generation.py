import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import json
import random
import argparse
from datetime import datetime
from tqdm import tqdm
import multiprocessing as mp
import time

# --- 1. 移动函数定义和类定义到模块顶层或一个可被导入的模块 ---
# (将您原文件中的所有辅助函数、模型类、get_data_loaders, train_model, evaluate_model 放在这里)
# ... (此处省略具体函数定义，直接复制粘贴您原文件对应部分) ...

# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 不同数据集的CNN架构
class MNISTCNN(nn.Module):
    def __init__(self, use_batchnorm=True, dropout_rate=0.5, activation='ReLU', init_type='he'):
        super(MNISTCNN, self).__init__()
        # 激活函数选择
        if activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'ReLU6':
            self.activation = nn.ReLU6()
        elif activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU(negative_slope=0.01)
        elif activation == 'PReLU':
            self.activation = nn.PReLU()
        elif activation == 'ELU':
            self.activation = nn.ELU(alpha=1.0)
        elif activation == 'GELU':
            self.activation = nn.GELU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.ReLU()

        # 卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # 批归一化层
        if use_batchnorm:
            self.bn1 = nn.BatchNorm2d(32)
            self.bn2 = nn.BatchNorm2d(64)
            self.bn3 = nn.BatchNorm2d(128)
        else:
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()
            self.bn3 = nn.Identity()

        # 池化层
        self.pool = nn.MaxPool2d(2, 2)

        # 全连接层
        self.fc1 = nn.Linear(128 * 7 * 7, 256) # 128 * 3 * 3 = 1152 (after 3 maxpool operations)
        self.fc2 = nn.Linear(256, 10)

        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)

        # 初始化权重
        self._initialize_weights(init_type)

    def _initialize_weights(self, init_type):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if init_type == 'he':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif init_type == 'xavier':
                    nn.init.xavier_uniform_(m.weight)
                elif init_type == 'zero':
                    nn.init.zeros_(m.weight)
                elif init_type == 'large':
                    nn.init.normal_(m.weight, mean=0, std=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                if init_type == 'he':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif init_type == 'xavier':
                    nn.init.xavier_uniform_(m.weight)
                elif init_type == 'zero':
                    nn.init.zeros_(m.weight)
                elif init_type == 'large':
                    nn.init.normal_(m.weight, mean=0, std=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.pool(self.activation(self.bn1(self.conv1(x))))
        x = self.pool(self.activation(self.bn2(self.conv2(x))))
        x = self.activation(self.bn3(self.conv3(x)))
        x = x.view(-1, 128 * 7 * 7)
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class CIFAR10CNN(nn.Module):
    def __init__(self, use_batchnorm=True, dropout_rate=0.5, activation='ReLU', init_type='he'):
        super(CIFAR10CNN, self).__init__()
        if activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'ReLU6':
            self.activation = nn.ReLU6()
        elif activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU(negative_slope=0.01)
        elif activation == 'PReLU':
            self.activation = nn.PReLU()
        elif activation == 'ELU':
            self.activation = nn.ELU(alpha=1.0)
        elif activation == 'GELU':
            self.activation = nn.GELU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.ReLU()

        # 更深的网络适应CIFAR-10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        if use_batchnorm:
            self.bn1 = nn.BatchNorm2d(32)
            self.bn2 = nn.BatchNorm2d(32)
            self.bn3 = nn.BatchNorm2d(64)
            self.bn4 = nn.BatchNorm2d(64)
            self.bn5 = nn.BatchNorm2d(128)
        else:
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()
            self.bn3 = nn.Identity()
            self.bn4 = nn.Identity()
            self.bn5 = nn.Identity()

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout_layer = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(128 * 4 * 4, 256) # CIFAR-10: 32x32 -> after 3 pools: 4x4
        self.fc2 = nn.Linear(256, 10)
        self._initialize_weights(init_type)

    def _initialize_weights(self, init_type):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if init_type == 'he':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif init_type == 'xavier':
                    nn.init.xavier_uniform_(m.weight)
                elif init_type == 'zero':
                    nn.init.zeros_(m.weight)
                elif init_type == 'large':
                    nn.init.normal_(m.weight, mean=0, std=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                if init_type == 'he':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif init_type == 'xavier':
                    nn.init.xavier_uniform_(m.weight)
                elif init_type == 'zero':
                    nn.init.zeros_(m.weight)
                elif init_type == 'large':
                    nn.init.normal_(m.weight, mean=0, std=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.pool(self.activation(self.bn1(self.conv1(x))))
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.pool(self.activation(self.bn3(self.conv3(x))))
        x = self.activation(self.bn4(self.conv4(x)))
        x = self.pool(self.activation(self.bn5(self.conv5(x))))
        x = x.view(-1, 128 * 4 * 4)
        x = self.activation(self.fc1(x))
        x = self.dropout_layer(x)
        x = self.fc2(x)
        return x

class SVHNCNN(nn.Module):
    def __init__(self, use_batchnorm=True, dropout_rate=0.5, activation='ReLU', init_type='he'):
        super(SVHNCNN, self).__init__()
        if activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'ReLU6':
            self.activation = nn.ReLU6()
        elif activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU(negative_slope=0.01)
        elif activation == 'PReLU':
            self.activation = nn.PReLU()
        elif activation == 'ELU':
            self.activation = nn.ELU(alpha=1.0)
        elif activation == 'GELU':
            self.activation = nn.GELU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.ReLU()

        # SVHN专用网络，稍微简化
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        if use_batchnorm:
            self.bn1 = nn.BatchNorm2d(32)
            self.bn2 = nn.BatchNorm2d(64)
            self.bn3 = nn.BatchNorm2d(128)
        else:
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()
            self.bn3 = nn.Identity()

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout_layer = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)
        self._initialize_weights(init_type)

    def _initialize_weights(self, init_type):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if init_type == 'he':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif init_type == 'xavier':
                    nn.init.xavier_uniform_(m.weight)
                elif init_type == 'zero':
                    nn.init.zeros_(m.weight)
                elif init_type == 'large':
                    nn.init.normal_(m.weight, mean=0, std=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                if init_type == 'he':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif init_type == 'xavier':
                    nn.init.xavier_uniform_(m.weight)
                elif init_type == 'zero':
                    nn.init.zeros_(m.weight)
                elif init_type == 'large':
                    nn.init.normal_(m.weight, mean=0, std=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.pool(self.activation(self.bn1(self.conv1(x))))
        x = self.pool(self.activation(self.bn2(self.conv2(x))))
        x = self.pool(self.activation(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 4 * 4)
        x = self.activation(self.fc1(x))
        x = self.dropout_layer(x)
        x = self.fc2(x)
        return x

class STL10CNN(nn.Module):
    def __init__(self, use_batchnorm=True, dropout_rate=0.5, activation='ReLU', init_type='he'):
        super(STL10CNN, self).__init__()
        if activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'ReLU6':
            self.activation = nn.ReLU6()
        elif activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU(negative_slope=0.01)
        elif activation == 'PReLU':
            self.activation = nn.PReLU()
        elif activation == 'ELU':
            self.activation = nn.ELU(alpha=1.0)
        elif activation == 'GELU':
            self.activation = nn.GELU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.ReLU()

        # STL-10专用网络，适应更高分辨率(96x96)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        if use_batchnorm:
            self.bn1 = nn.BatchNorm2d(64)
            self.bn2 = nn.BatchNorm2d(128)
            self.bn3 = nn.BatchNorm2d(256)
            self.bn4 = nn.BatchNorm2d(256)
        else:
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()
            self.bn3 = nn.Identity()
            self.bn4 = nn.Identity()

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout_layer = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(256 * 6 * 6, 512) # STL-10: 96x96 -> after 4 pools: 6x6
        self.fc2 = nn.Linear(512, 10)
        self._initialize_weights(init_type)

    def _initialize_weights(self, init_type):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if init_type == 'he':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif init_type == 'xavier':
                    nn.init.xavier_uniform_(m.weight)
                elif init_type == 'zero':
                    nn.init.zeros_(m.weight)
                elif init_type == 'large':
                    nn.init.normal_(m.weight, mean=0, std=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                if init_type == 'he':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif init_type == 'xavier':
                    nn.init.xavier_uniform_(m.weight)
                elif init_type == 'zero':
                    nn.init.zeros_(m.weight)
                elif init_type == 'large':
                    nn.init.normal_(m.weight, mean=0, std=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.pool(self.activation(self.bn1(self.conv1(x))))
        x = self.pool(self.activation(self.bn2(self.conv2(x))))
        x = self.pool(self.activation(self.bn3(self.conv3(x))))
        x = self.pool(self.activation(self.bn4(self.conv4(x))))
        x = x.view(-1, 256 * 6 * 6)
        x = self.activation(self.fc1(x))
        x = self.dropout_layer(x)
        x = self.fc2(x)
        return x

class FashionMNISTCNN(nn.Module):
    def __init__(self, use_batchnorm=True, dropout_rate=0.5, activation='ReLU', init_type='he'):
        super(FashionMNISTCNN, self).__init__()
        if activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'ReLU6':
            self.activation = nn.ReLU6()
        elif activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU(negative_slope=0.01)
        elif activation == 'PReLU':
            self.activation = nn.PReLU()
        elif activation == 'ELU':
            self.activation = nn.ELU(alpha=1.0)
        elif activation == 'GELU':
            self.activation = nn.GELU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.ReLU()

        # Fashion-MNIST专用网络，比MNIST稍复杂
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        if use_batchnorm:
            self.bn1 = nn.BatchNorm2d(32)
            self.bn2 = nn.BatchNorm2d(64)
            self.bn3 = nn.BatchNorm2d(128)
        else:
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()
            self.bn3 = nn.Identity()

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout_layer = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(128 * 3 * 3, 256) # Fashion-MNIST: 28x28 -> after 3 pools: 3x3
        self.fc2 = nn.Linear(256, 10)
        self._initialize_weights(init_type)

    def _initialize_weights(self, init_type):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if init_type == 'he':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif init_type == 'xavier':
                    nn.init.xavier_uniform_(m.weight)
                elif init_type == 'zero':
                    nn.init.zeros_(m.weight)
                elif init_type == 'large':
                    nn.init.normal_(m.weight, mean=0, std=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                if init_type == 'he':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif init_type == 'xavier':
                    nn.init.xavier_uniform_(m.weight)
                elif init_type == 'zero':
                    nn.init.zeros_(m.weight)
                elif init_type == 'large':
                    nn.init.normal_(m.weight, mean=0, std=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.pool(self.activation(self.bn1(self.conv1(x))))
        x = self.pool(self.activation(self.bn2(self.conv2(x))))
        x = self.pool(self.activation(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 3 * 3)
        x = self.activation(self.fc1(x))
        x = self.dropout_layer(x)
        x = self.fc2(x)
        return x

def get_data_loaders(dataset_name, batch_size, data_augmentation=False, label_noise_ratio=0.0, data_subset_ratio=1.0):
    """获取指定数据集的数据加载器"""
    # 数据增强变换
    if dataset_name in ['cifar10', 'svhn', 'stl10']:
        if data_augmentation:
            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.RandomCrop(32, padding=4) if dataset_name != 'stl10' else transforms.Resize((96, 96)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # STL-10特殊处理
        if dataset_name == 'stl10':
            if data_augmentation:
                train_transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=10),
                    transforms.Resize((96, 96)),
                    transforms.RandomCrop(96, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
            else:
                train_transform = transforms.Compose([
                    transforms.Resize((96, 96)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
            test_transform = transforms.Compose([
                transforms.Resize((96, 96)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    elif dataset_name == 'fashion_mnist':
        if data_augmentation:
            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,))
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,))
            ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
    elif dataset_name == 'mnist':
        # <-- 新增：MNIST 专用处理
        if data_augmentation:
            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # 平移增强
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)) # MNIST 官方归一化参数
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    # 加载数据集
    if dataset_name == 'cifar10':
        full_train_dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=train_transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=test_transform
        )
    elif dataset_name == 'svhn':
        full_train_dataset = torchvision.datasets.SVHN(
            root='./data',
            split='train',
            download=True,
            transform=train_transform
        )
        test_dataset = torchvision.datasets.SVHN(
            root='./data',
            split='test',
            download=True,
            transform=test_transform
        )
    elif dataset_name == 'stl10':
        full_train_dataset = torchvision.datasets.STL10(
            root='./data',
            split='train',
            download=True,
            transform=train_transform
        )
        test_dataset = torchvision.datasets.STL10(
            root='./data',
            split='test',
            download=True,
            transform=test_transform
        )
    elif dataset_name == 'fashion_mnist':
        full_train_dataset = torchvision.datasets.FashionMNIST(
            root='./data',
            train=True,
            download=True,
            transform=train_transform
        )
        test_dataset = torchvision.datasets.FashionMNIST(
            root='./data',
            train=False,
            download=True,
            transform=test_transform
        )
    elif dataset_name == 'mnist':
        # <-- 新增：加载 MNIST
        full_train_dataset = torchvision.datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=train_transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root='./data',
            train=False,
            download=True,
            transform=test_transform
        )

    # 创建验证集
    indices = list(range(len(full_train_dataset)))
    np.random.shuffle(indices)
    if dataset_name == 'stl10':
        val_size = min(1000, len(full_train_dataset)) # STL-10可能较小
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
    else:
        val_size = min(5000, len(full_train_dataset))
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

    # 应用标签噪声
    if label_noise_ratio > 0:
        for i in train_indices:
            if random.random() < label_noise_ratio:
                if dataset_name in ['stl10', 'svhn']:
                    full_train_dataset.labels[i] = random.randint(0, 9)
                else:
                    full_train_dataset.targets[i] = random.randint(0, 9)

    # 应用数据子集
    if data_subset_ratio < 1.0:
        num_samples = int(len(train_indices) * data_subset_ratio)
        train_indices = train_indices[:num_samples]

    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_train_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, save_checkpoints=None, dataset_name=None, seed=None):
    """训练模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    best_val_acc = 0.0
    patience_counter = 0
    history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100. * correct / total
        train_loss = running_loss / len(train_loader)

        # 验证阶段
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total
        val_loss = val_running_loss / len(val_loader)

        # 记录历史
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f'Epoch [{epoch+1}/{epochs}] Train Acc: {train_acc:.2f}% Val Acc: {val_acc:.2f}%')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

        # 保存检查点
        if save_checkpoints and epoch + 1 in save_checkpoints:
            checkpoint_path = f"checkpoints/{dataset_name}_seed{seed}_checkpoint_epoch_{epoch + 1}.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'history': history
            }, checkpoint_path)

    # 返回最佳模型
    model.load_state_dict(best_model_state)
    return model, history

def evaluate_model(model, dataloader):
    """评估模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total
    avg_loss = running_loss / len(dataloader)
    return accuracy, avg_loss

# --- 2. 定义单个任务的工作函数 ---
def run_single_task(task_args):
    """
    这个函数将在每个进程中运行一次。
    task_args 包含完成一个权重生成任务所需的所有参数。
    """
    # 解包任务参数
    dataset_info = task_args['dataset_info']
    model_class = task_args['model_class']
    task_config = task_args['task_config']
    task_index = task_args['task_index'] # 用于标识进程和日志

    dataset_name = dataset_info['name']
    type_key = task_config['type_key']
    sub_type = task_config['sub_type']
    start_idx = task_config['start_idx']
    end_idx = task_config['end_idx']
    config_details = task_config['config']

    print(f"[Process-{os.getpid()}-Task-{task_index}] Starting {type_key} ({sub_type}) tasks for {dataset_name}, indices {start_idx} to {end_idx - 1}")

    os.makedirs(f"weights/{dataset_name}", exist_ok=True)
    os.makedirs(f"checkpoints", exist_ok=True)

    config_records = []
    ACTIVATION_MAP = {
        'ReLU': nn.ReLU(), 'ReLU6': nn.ReLU6(), 'LeakyReLU': nn.LeakyReLU(negative_slope=0.01),
        'PReLU': nn.PReLU(), 'ELU': nn.ELU(alpha=1.0), 'GELU': nn.GELU()
    }
    available_optimizers = ['Adam', 'SGD']
    available_init_type = ['He', 'Xavier']
    available_data_augmentation_enabled = ['0', '1']

    for i in tqdm(range(start_idx, end_idx), desc=f"{dataset_name} {sub_type} Task-{task_index}"):
        # 训练模型
        #高质量训练快照
        if 'snapshot' in sub_type:
            seed = config_details['start_seed'] + i
            set_seed(seed)

            chosen_lr = config_details.get('learning_rate', 0.01)
            chosen_use_bn = config_details.get('use_batchnorm', True)
            chosen_dropout = config_details.get('dropout_rate', 0.5)
            chosen_batch_size = config_details.get('batch_size', 128)
            chosen_wd = config_details.get('weight_decay', 1e-4)
            chosen_init_type = random.choice(available_init_type)
            chosen_activation_name = random.choice(list(ACTIVATION_MAP.keys()))
            chosen_activation = ACTIVATION_MAP[chosen_activation_name]
            chosen_da = random.choice(available_data_augmentation_enabled)
            chosen_optimizer = random.choice(available_optimizers)

            # 构建模型
            model = model_class(
                use_batchnorm=chosen_use_bn,
                dropout_rate=chosen_dropout,
                activation=chosen_activation, # 注意这里传递的是字符串名，因为初始化时需要
                init_type=chosen_init_type # 确保init_type是小写，与初始化函数一致
            )

            # 获取数据加载器
            train_loader, val_loader, test_loader = get_data_loaders(
                dataset_name, 
                batch_size=chosen_batch_size, 
                data_augmentation=chosen_da
            )

            # 设置优化器
            if chosen_optimizer == 'Adam':
                optimizer = optim.Adam(
                    model.parameters(),
                    lr=chosen_lr, 
                    weight_decay=chosen_wd
                )
            elif chosen_optimizer == 'SGD':
                optimizer = optim.SGD(
                    model.parameters(), 
                    lr=chosen_lr, 
                    weight_decay=chosen_wd, 
                    momentum=0.9, 
                    nesterov=True
                )

            criterion = nn.CrossEntropyLoss()
            
            snapshot_epochs = [4, 5, 6]
            checkpoint_dir = "checkpoints"
            trained_model, history = train_model(
                model=model, 
                train_loader=train_loader, 
                val_loader=val_loader,
                optimizer=optimizer, 
                criterion=criterion, 
                epochs=max(snapshot_epochs),
                save_checkpoints=snapshot_epochs, 
                dataset_name=dataset_name, 
                seed=seed
            )
            # 从检查点加载并保存每个epoch的快照
            for epoch in snapshot_epochs:
                checkpoint_path = f"checkpoints/{dataset_name}_seed{seed}_checkpoint_epoch_{epoch}.pth"
                try:
                    checkpoint = torch.load(checkpoint_path, map_location='cpu')
                    temp_model = model_class(
                        use_batchnorm=chosen_use_bn,
                        dropout_rate=chosen_dropout,
                        activation=chosen_activation,
                        init_type=chosen_init_type
                    )
                    temp_model.load_state_dict(checkpoint['model_state_dict'])
                    temp_model.eval()
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    temp_model = temp_model.to(device)

                    val_acc, val_loss = evaluate_model(temp_model, val_loader)
                    test_acc, test_loss = evaluate_model(temp_model, test_loader)

                    weight_path = f"weights/{dataset_name}/{dataset_name}_good_snapshot_seed{seed:03d}_epoch{epoch:03d}.pth"
                    torch.save({
                        'model_state_dict': checkpoint['model_state_dict'],
                        'seed': seed,
                        'hyperparameters': {
                            'learning_rate': chosen_lr, 
                            'optimizer': chosen_optimizer,
                            'batch_size': chosen_batch_size, 
                            'init_type': chosen_init_type,
                            'use_batchnorm': chosen_use_bn, 
                            'activation': chosen_activation_name,
                            'data_augmentation': chosen_da, 
                            'epochs': epoch, 
                            'weight_decay': chosen_wd
                        },
                        'performance': {
                            'val_accuracy': val_acc, 
                            'val_loss': val_loss,
                            'test_accuracy': test_acc, 
                            'test_loss': test_loss,
                            'training_history_up_to_epoch': {k: v[:epoch] for k, v in history.items()}
                        }
                    }, weight_path)
                    config_records.append({
                        'weight_id': f'{dataset_name}_good_snapshot_seed{seed:03d}_epoch{epoch:03d}',
                        'path': weight_path, 
                        'type': 'high_quality_snapshot', 
                        'seed': seed,
                        'epoch': epoch,
                        'hyperparameters': {
                            'learning_rate': chosen_lr, 
                            'optimizer': chosen_optimizer,
                            'batch_size': chosen_batch_size, 
                            'init_type': chosen_init_type,
                            'use_batchnorm': chosen_use_bn, 
                            'activation': chosen_activation_name,
                            'data_augmentation': chosen_da, 
                            'epochs': epoch,
                            'weight_decay': chosen_wd
                        },
                        'performance': {
                            'val_accuracy': val_acc, 
                            'val_loss': val_loss,
                            'test_accuracy': test_acc, 
                            'test_loss': test_loss,
                            'training_history_up_to_epoch': {k: v[:epoch] for k, v in history.items()}
                        }
                    })
                except FileNotFoundError:
                    print(f"[Warning] Checkpoint {checkpoint_path} not found, skipping.")

        #低质量训练快照
        elif 'undertrained' in sub_type:
            seed = config_details['start_seed'] + i
            set_seed(seed)

            #chosen_lr = config_details.get('learning_rate', 0.01)
            chosen_lr = 0.003
            chosen_use_bn = config_details.get('use_batchnorm', True)
            chosen_dropout = config_details.get('dropout_rate', 0.5)
            chosen_batch_size = config_details.get('batch_size', 128)
            chosen_wd = config_details.get('weight_decay', 1e-4)
            chosen_init_type = random.choice(available_init_type)
            chosen_activation_name = random.choice(list(ACTIVATION_MAP.keys()))
            chosen_activation = ACTIVATION_MAP[chosen_activation_name]
            chosen_da = False
            chosen_optimizer = random.choice(available_optimizers)

            # 构建模型
            model = model_class(
                use_batchnorm=chosen_use_bn,
                dropout_rate=chosen_dropout,
                activation=chosen_activation, # 注意这里传递的是字符串名，因为初始化时需要
                init_type=chosen_init_type # 确保init_type是小写，与初始化函数一致
            )

            # 获取数据加载器
            train_loader, val_loader, test_loader = get_data_loaders(
                dataset_name, 
                batch_size=chosen_batch_size, 
                data_augmentation=chosen_da
            )

            # 设置优化器
            if chosen_optimizer == 'Adam':
                optimizer = optim.Adam(
                    model.parameters(),
                    lr=chosen_lr, 
                    weight_decay=chosen_wd
                )
            elif chosen_optimizer == 'SGD':
                optimizer = optim.SGD(
                    model.parameters(), 
                    lr=chosen_lr, 
                    weight_decay=chosen_wd, 
                    momentum=0.9, 
                    nesterov=True
                )

            criterion = nn.CrossEntropyLoss()
            low_epochs = [3, 4, 5]
            trained_model, history = train_model(
                model=model, 
                train_loader=train_loader, 
                val_loader=val_loader,
                optimizer=optimizer, 
                criterion=criterion, 
                epochs=max(low_epochs),
                save_checkpoints=low_epochs, 
                dataset_name=dataset_name, 
                seed=seed
            )
            for epoch in low_epochs:
                checkpoint_path = f"checkpoints/{dataset_name}_seed{seed}_checkpoint_epoch_{epoch}.pth"
                try:
                    checkpoint = torch.load(checkpoint_path, map_location='cpu')
                    temp_model = model_class(
                        use_batchnorm=chosen_use_bn,
                        dropout_rate=chosen_dropout,
                        activation=chosen_activation,
                        init_type=chosen_init_type
                    )
                    temp_model.load_state_dict(checkpoint['model_state_dict'])
                    temp_model.eval()
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    temp_model = temp_model.to(device)

                    val_acc, val_loss = evaluate_model(temp_model, val_loader)
                    test_acc, test_loss = evaluate_model(temp_model, test_loader)

                    weight_path = f"weights/{dataset_name}/{dataset_name}_bad_undertrained_seed_{seed:03d}_epoch_{epoch:03d}.pth"
                    torch.save({
                        'model_state_dict': checkpoint['model_state_dict'],
                        'seed': seed,
                        'hyperparameters': {
                            'learning_rate': chosen_lr, 
                            'optimizer': chosen_optimizer,
                            'batch_size': chosen_batch_size, 
                            'init_type': chosen_init_type,
                            'use_batchnorm': chosen_use_bn, 
                            'activation': chosen_activation_name,
                            'data_augmentation': False, 
                            'epochs': epoch, 
                            'weight_decay': chosen_wd
                        },
                        'performance': {
                            'val_accuracy': val_acc, 
                            'val_loss': val_loss,
                            'test_accuracy': test_acc, 
                            'test_loss': test_loss,
                            'training_history_up_to_epoch': {k: v[:epoch] for k, v in history.items()}
                        }
                    }, weight_path)
                    config_records.append({
                        'weight_id': f'{dataset_name}_bad_undertrained_seed_{seed:03d}_epoch_{epoch:03d}',
                        'path': weight_path, 
                        'type': 'low_quality_undertrained', 
                        'seed': seed,
                        'epoch': epoch,
                        'hyperparameters': {
                            'learning_rate': chosen_lr, 
                            'optimizer': chosen_optimizer,
                            'batch_size': chosen_batch_size, 
                            'init_type': chosen_init_type,
                            'use_batchnorm': chosen_use_bn, 
                            'activation': chosen_activation_name,
                            'data_augmentation': False, 
                            'epochs': epoch,
                            'weight_decay': chosen_wd
                        },
                        'performance': {
                            'val_accuracy': val_acc, 
                            'val_loss': val_loss,
                            'test_accuracy': test_acc, 
                            'test_loss': test_loss,
                            'training_history_up_to_epoch': {k: v[:epoch] for k, v in history.items()}
                        }
                    })
                except FileNotFoundError:
                    print(f"[Warning] Checkpoint {checkpoint_path} not found, skipping.")

        #高质量学习率变体
        elif 'lr_variant' in sub_type:
            seed = config_details['start_seed'] + i
            set_seed(seed)

            chosen_use_bn = config_details.get('use_batchnorm', True)
            chosen_dropout = config_details.get('dropout_rate', 0.5)
            chosen_batch_size = config_details.get('batch_size', 128)
            chosen_wd = config_details.get('weight_decay', 1e-4)
            chosen_init_type = random.choice(available_init_type)
            chosen_activation_name = random.choice(list(ACTIVATION_MAP.keys()))
            chosen_activation = ACTIVATION_MAP[chosen_activation_name]
            chosen_da = random.choice(available_data_augmentation_enabled)
            chosen_optimizer = random.choice(available_optimizers)
            chosen_epochs = 10

            lr_configs = [0.001, 0.002, 0.004]

            chosen_lr = lr_configs[(i - start_idx) % len(lr_configs)]

            # 构建模型
            model = model_class(
                use_batchnorm=chosen_use_bn,
                dropout_rate=chosen_dropout,
                activation=chosen_activation, # 注意这里传递的是字符串名，因为初始化时需要
                init_type=chosen_init_type # 确保init_type是小写，与初始化函数一致
            )

            # 获取数据加载器
            train_loader, val_loader, test_loader = get_data_loaders(
                dataset_name, 
                batch_size=chosen_batch_size, 
                data_augmentation=chosen_da
            )

            if chosen_optimizer == 'Adam':
                optimizer = optim.Adam(
                    model.parameters(), 
                    lr=chosen_lr, 
                    weight_decay=chosen_wd
                ) 
            else:
                optimizer = optim.SGD(
                    model.parameters(), 
                    lr=chosen_lr, 
                    weight_decay=chosen_wd, 
                    momentum=0.9, 
                    nesterov=True
                )
            criterion = nn.CrossEntropyLoss()

            trained_model, history = train_model(
                model=model, 
                train_loader=train_loader, 
                val_loader=val_loader,
                optimizer=optimizer, 
                criterion=criterion, 
                epochs=chosen_epochs,
                dataset_name=dataset_name, 
                seed=seed
            )
            val_acc, val_loss = evaluate_model(trained_model, val_loader)
            test_acc, test_loss = evaluate_model(trained_model, test_loader)

            weight_path = f"weights/{dataset_name}/{dataset_name}_good_lr_variant_{i+1:03d}.pth"
            torch.save({
                'model_state_dict': trained_model.state_dict(),
                'seed': seed,
                'hyperparameters': {
                    'learning_rate': chosen_lr, 
                    'optimizer': chosen_optimizer,
                    'batch_size': chosen_batch_size, 
                    'init_type': chosen_init_type,
                    'use_batchnorm': chosen_use_bn, 
                    'activation': chosen_activation_name,
                    'data_augmentation': chosen_da, 
                    'epochs': chosen_epochs, 
                    'weight_decay': chosen_wd,
                },
                'performance': {
                    'val_accuracy': val_acc, 
                    'val_loss': val_loss,
                    'test_accuracy': test_acc, 
                    'test_loss': test_loss,
                    'training_history': history
                }
            }, weight_path)
            config_records.append({
                'weight_id': f'{dataset_name}_good_lr_variant_{i+1:03d}',
                'path': weight_path, 
                'type': 'high_quality_lr_variant', 
                'seed': seed,
                'hyperparameters': {
                    'learning_rate': chosen_lr, 
                    'optimizer': chosen_optimizer,
                    'batch_size': chosen_batch_size, 
                    'init_type': chosen_init_type,
                    'use_batchnorm': chosen_use_bn, 
                    'activation': chosen_activation_name,
                    'data_augmentation': chosen_da, 
                    'epochs': chosen_epochs, 
                    'weight_decay': chosen_wd,
                },
                'performance': {
                    'val_accuracy': val_acc, 
                    'val_loss': val_loss,
                    'test_accuracy': test_acc, 
                    'test_loss': test_loss,
                    'training_history': history
                }
            })
        
        #低质量数据问题
        elif 'data_issue' in sub_type:
            seed = config_details['start_seed'] + i
            set_seed(seed)

            #chosen_lr = config_details.get('learning_rate', 0.01)
            chosen_lr = 0.003
            chosen_use_bn = config_details.get('use_batchnorm', True)
            chosen_dropout = config_details.get('dropout_rate', 0.5)
            chosen_batch_size = config_details.get('batch_size', 128)
            chosen_wd = config_details.get('weight_decay', 1e-4)
            chosen_init_type = random.choice(available_init_type)
            chosen_activation_name = random.choice(list(ACTIVATION_MAP.keys()))
            chosen_activation = ACTIVATION_MAP[chosen_activation_name]
            chosen_da = False
            chosen_optimizer = random.choice(available_optimizers)
            #chosen_epochs = config_details.get('epochs')
            chosen_epochs = 10

            model = model_class(
                use_batchnorm=chosen_use_bn,
                dropout_rate=chosen_dropout,
                activation=chosen_activation, # 注意这里传递的是字符串名，因为初始化时需要
                init_type=chosen_init_type # 确保init_type是小写，与初始化函数一致
            )

            label_noise = 0.2
            train_loader, val_loader, test_loader = get_data_loaders(
                dataset_name, 
                batch_size=chosen_batch_size, 
                data_augmentation=chosen_da, 
                label_noise_ratio=label_noise
            )

            if chosen_optimizer == 'Adam':
                optimizer = optim.Adam(
                    model.parameters(),
                    lr=chosen_lr, 
                    weight_decay=chosen_wd
                )
            else:
                optimizer = optim.SGD(
                    model.parameters(), 
                    lr=chosen_lr, 
                    weight_decay=chosen_wd, 
                    momentum=0.9, 
                    nesterov=True
                )

            criterion = nn.CrossEntropyLoss()

            trained_model, history = train_model(
                model=model, 
                train_loader=train_loader, 
                val_loader=val_loader,
                optimizer=optimizer, 
                criterion=criterion, 
                epochs=chosen_epochs, 
                dataset_name=dataset_name, 
            )

            val_acc, val_loss = evaluate_model(trained_model, val_loader)
            test_acc, test_loss = evaluate_model(trained_model, test_loader)

            weight_path = f"weights/{dataset_name}/{dataset_name}_bad_data_issue_{i+1:03d}.pth"
            torch.save({
                'model_state_dict': trained_model.state_dict(),
                'seed': seed,
                'hyperparameters': {
                    'learning_rate': chosen_lr, 
                    'optimizer': chosen_optimizer,
                    'batch_size': chosen_batch_size, 
                    'init_type': chosen_init_type,
                    'use_batchnorm': chosen_use_bn, 
                    'activation': chosen_activation_name,
                    'data_augmentation': chosen_da, 
                    'epochs': chosen_epochs, 
                    'weight_decay': chosen_wd,
                },
                'performance': {
                    'val_accuracy': val_acc, 
                    'val_loss': val_loss,
                    'test_accuracy': test_acc, 
                    'test_loss': test_loss,
                    'training_history': history
                }
            }, weight_path)
            config_records.append({
                'weight_id': f'{dataset_name}_bad_data_issue_{i+1:03d}',
                'path': weight_path, 
                'type': 'low_quality_data_issue', 
                'seed': seed,
                'hyperparameters': {
                    'learning_rate': chosen_lr, 
                    'optimizer': chosen_optimizer,
                    'batch_size': chosen_batch_size, 
                    'init_type': chosen_init_type,
                    'use_batchnorm': chosen_use_bn, 
                    'activation': chosen_activation_name,
                    'data_augmentation': chosen_da, 
                    'epochs': chosen_epochs, 
                    'weight_decay': chosen_wd,
                },
                'performance': {
                    'val_accuracy': val_acc, 
                    'val_loss': val_loss,
                    'test_accuracy': test_acc, 
                    'test_loss': test_loss,
                    'training_history': history
                }
            })

        #生成高质量权重
        else:
            seed = config_details['start_seed'] + i
            set_seed(seed)

            chosen_lr = config_details.get('learning_rate', 0.01)
            chosen_use_bn = config_details.get('use_batchnorm', True)
            chosen_dropout = config_details.get('dropout_rate', 0.5)
            chosen_batch_size = config_details.get('batch_size', 128)
            chosen_wd = config_details.get('weight_decay', 1e-4)
            chosen_init_type = random.choice(available_init_type)
            chosen_activation_name = random.choice(list(ACTIVATION_MAP.keys()))
            chosen_activation = ACTIVATION_MAP[chosen_activation_name]
            chosen_da = random.choice(available_data_augmentation_enabled)
            chosen_optimizer = random.choice(available_optimizers)
            # chosen_epochs = config_details.get('epochs')
            chosen_epochs = 10

            model = model_class(
                use_batchnorm=chosen_use_bn,
                dropout_rate=chosen_dropout,
                activation=chosen_activation, # 注意这里传递的是字符串名，因为初始化时需要
                init_type=chosen_init_type # 确保init_type是小写，与初始化函数一致
            )

            train_loader, val_loader, test_loader = get_data_loaders(
                dataset_name, 
                batch_size=chosen_batch_size, 
                data_augmentation=chosen_da, 
            )

            if chosen_optimizer == 'Adam':
                optimizer = optim.Adam(
                    model.parameters(),
                    lr=chosen_lr, 
                    weight_decay=chosen_wd
                )
            elif chosen_optimizer == 'SGD':
                optimizer = optim.SGD(
                    model.parameters(), 
                    lr=chosen_lr, 
                    weight_decay=chosen_wd, 
                    momentum=0.9, 
                    nesterov=True
                )

            criterion = nn.CrossEntropyLoss()

            trained_model, history = train_model(
                model=model, 
                train_loader=train_loader, 
                val_loader=val_loader,
                optimizer=optimizer, 
                criterion=criterion, 
                epochs=chosen_epochs, 
                dataset_name=dataset_name, 
                seed=seed
            )

            val_acc, val_loss = evaluate_model(trained_model, val_loader)
            test_acc, test_loss = evaluate_model(trained_model, test_loader)

            weight_path = f"weights/{dataset_name}/{dataset_name}_good_standard_{i+1:03d}.pth"
            torch.save({
                'model_state_dict': trained_model.state_dict(),
                'seed': seed,
                'hyperparameters': {
                    'learning_rate': chosen_lr, 
                    'optimizer': chosen_optimizer,
                    'batch_size': chosen_batch_size, 
                    'init_type': chosen_init_type,
                    'use_batchnorm': chosen_use_bn, 
                    'activation': chosen_activation_name,
                    'data_augmentation': chosen_da, 
                    'epochs': chosen_epochs, 
                    'weight_decay': chosen_wd,
                },
                'performance': {
                    'val_accuracy': val_acc,
                    'val_loss': val_loss,
                    'test_accuracy': test_acc,
                    'test_loss': test_loss,
                    'training_history': history
                }
            }, weight_path)
        
            config_records.append({
                'weight_id': f'{dataset_name}_good_standard_{i+1:03d}',
                'path': weight_path,
                'type': 'high_quality_standard',
                'seed': seed,
                'hyperparameters': {
                    'learning_rate': chosen_lr, 
                    'optimizer': chosen_optimizer,
                    'batch_size': chosen_batch_size, 
                    'init_type': chosen_init_type,
                    'use_batchnorm': chosen_use_bn, 
                    'activation': chosen_activation_name,
                    'data_augmentation': chosen_da, 
                    'epochs': chosen_epochs, 
                    'weight_decay': chosen_wd,
                },
                'performance': {
                    'val_accuracy': val_acc,
                    'val_loss': val_loss,
                    'test_accuracy': test_acc,
                    'test_loss': test_loss,
                    'training_history': history
                }
            })
                
    print(f"[Process-{os.getpid()}-Task-{task_index}] Finished {type_key} ({sub_type}) tasks for {dataset_name}")
    return config_records 


# --- 3. 主函数 ---
def main():
    print("程序开始！！！！！！！！！！！")
    parser = argparse.ArgumentParser(description="Parallel Weight Generation")
    parser.add_argument('--processes', type=int, default=50, help='Number of processes to use (default: 50)')
    args = parser.parse_args()

    num_processes = args.processes
    print(f"Starting parallel weight generation with {num_processes} processes...")

    # --- 定义数据集配置 ---
    datasets_info = [
        {
            'name': 'svhn',
            'model_class': SVHNCNN,
            'high_quality_config': {
                'start_seed': 1, 
                'learning_rate': 0.001, 
                'batch_size': 128,
                'dropout_rate': 0.5, 
                'use_batchnorm': True, 
                'epochs': 20, 
                'weight_decay': 1e-4
            },
            'low_quality_config': {
                'start_seed': 1001, 
                'learning_rate': 0.001, 
                'batch_size': 128,
                'dropout_rate': 0.0, 
                'use_batchnorm': False, 
                'epochs': 10, 
                'weight_decay': 1e-4
            }
        },
    ]

    all_tasks = []
    task_counter = 0
    for ds_info in datasets_info:
        dataset_name = ds_info['name']
        model_class = ds_info['model_class']
        hq_config = ds_info['high_quality_config']
        lq_config = ds_info['low_quality_config']

        # 高质量任务
        # 1. 标准训练 (500个)
        tasks_per_process_hq_std = 500 // num_processes
        remainder_hq_std = 500 % num_processes
        start_idx = 0
        for p_idx in range(num_processes):
            end_idx = start_idx + tasks_per_process_hq_std
            if p_idx < remainder_hq_std:
                end_idx += 1
            if end_idx > start_idx: # Only add task if there's work to do
                all_tasks.append({
                    'dataset_info': ds_info,
                    'model_class': model_class,
                    'task_config': {
                        'type_key': 'high_quality',
                        'sub_type': 'standard',
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'config': hq_config
                    },
                    'task_index': task_counter
                })
                task_counter += 1
                start_idx = end_idx

        # 2. 快照 (300个) - 需要分发到不同进程
        # 由于快照是从特定epoch保存的，我们可以先启动基础训练任务，然后让它们自己保存快照
        # 为了简单，我们把这300个快照任务也均匀分配
        tasks_per_process_hq_snap = 100 // num_processes
        remainder_hq_snap = 100 % num_processes
        start_idx = 500 # Start from where standard left off for seeds
        for p_idx in range(num_processes):
            end_idx = start_idx + tasks_per_process_hq_snap
            if p_idx < remainder_hq_snap:
                end_idx += 1
            if end_idx > start_idx:
                all_tasks.append({
                    'dataset_info': ds_info,
                    'model_class': model_class,
                    'task_config': {
                        'type_key': 'high_quality',
                        'sub_type': 'snapshot',
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'config': hq_config # Use same config, but internally handles snapshots
                    },
                    'task_index': task_counter
                })
                task_counter += 1
                start_idx = end_idx

        # 3. 变体 (200个)
        tasks_per_process_hq_var = 200 // num_processes
        remainder_hq_var = 200 % num_processes
        start_idx = 600 # Start from where snapshot left off for seeds
        for p_idx in range(num_processes):
            end_idx = start_idx + tasks_per_process_hq_var
            if p_idx < remainder_hq_var:
                end_idx += 1
            if end_idx > start_idx:
                all_tasks.append({
                    'dataset_info': ds_info,
                    'model_class': model_class,
                    'task_config': {
                        'type_key': 'high_quality',
                        'sub_type': 'lr_variant',
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'config': hq_config
                    },
                    'task_index': task_counter
                })
                task_counter += 1
                start_idx = end_idx

        # 低质量任务
        # 1. 欠训练 (600个)
        tasks_per_process_lq_under = 200 // num_processes
        remainder_lq_under = 200 % num_processes
        start_idx = 800 # Low quality seed starts at 1001
        for p_idx in range(num_processes):
            end_idx = start_idx + tasks_per_process_lq_under
            if p_idx < remainder_lq_under:
                end_idx += 1
            if end_idx > start_idx:
                all_tasks.append({
                    'dataset_info': ds_info,
                    'model_class': model_class,
                    'task_config': {
                        'type_key': 'low_quality',
                        'sub_type': 'undertrained',
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'config': lq_config
                    },
                    'task_index': task_counter
                })
                task_counter += 1
                start_idx = end_idx

        # 2. 数据问题 (400个)
        tasks_per_process_lq_data = 400 // num_processes
        remainder_lq_data = 400 % num_processes
        start_idx = 1000 # Start from where undertrained left off for seeds
        for p_idx in range(num_processes):
            end_idx = start_idx + tasks_per_process_lq_data
            if p_idx < remainder_lq_data:
                end_idx += 1
            if end_idx > start_idx:
                all_tasks.append({
                    'dataset_info': ds_info,
                    'model_class': model_class,
                    'task_config': {
                        'type_key': 'low_quality',
                        'sub_type': 'data_issue',
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'config': lq_config
                    },
                    'task_index': task_counter
                })
                task_counter += 1
                start_idx = end_idx

    print(f"Total tasks to distribute: {len(all_tasks)} across {num_processes} processes.")

    # --- 使用进程池执行任务 ---
    with mp.Pool(processes=num_processes) as pool:
        # 提交所有任务
        results = pool.map(run_single_task, all_tasks)

    # --- 合并结果 ---
    all_configs = []
    for result_list in results:
        all_configs.extend(result_list)

    # --- 保存总体配置 ---
    with open('all_multi_dataset_weights_config_parallel.json', 'w') as f:
        json.dump(all_configs, f, indent=2)

    print(f"\nParallel generation completed!")
    print(f"Total weights generated: {len(all_configs)}")
    print(f"Configuration saved to 'all_multi_dataset_weights_config_parallel.json'")

    # 统计信息
    dataset_counts = {}
    type_counts = {}
    for config in all_configs:
        dataset = config['weight_id'].split('_')[0]
        w_type = config['type']
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
        type_counts[w_type] = type_counts.get(w_type, 0) + 1

    print("\nDataset Statistics:")
    for dataset, count in dataset_counts.items():
        print(f"  {dataset}: {count}")

    print("\nType Statistics:")
    for w_type, count in type_counts.items():
        print(f"  {w_type}: {count}")


if __name__ == "__main__":
    # 必须包含这个，对于Windows系统尤其重要
    mp.set_start_method('spawn', force=True)
    main()