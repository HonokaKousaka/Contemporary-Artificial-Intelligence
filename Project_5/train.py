# 引入必要的python库
import numpy as np
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertConfig, BertModel
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# 训练的过程
def train_model(model, epoch_num, optimizer, train_dataloader, valid_dataloader, train_count, valid_count, device):
    # 交叉熵损失函数
    loss_c = nn.CrossEntropyLoss()
    # 训练与验证的准确率
    train_acc = []
    valid_acc = []
    loss_array = []
    for epoch in range(epoch_num):
        loss = 0.0
        # 训练集和验证集正确判读的个数
        train_cor_count = 0
        valid_cor_count = 0
        for _, (img, des, target, idx, mask) in enumerate(train_dataloader):
            img, mask, idx, target = img.to(device), mask.to(device), idx.to(device), target.to(device)
            output = model(idx, mask, img)
            optimizer.zero_grad()
            loss = loss_c(output, target)
            # 反向传播
            loss.backward()
            optimizer.step()
            pred = output.argmax(dim=1)
            train_cor_count += int(pred.eq(target).sum())
            if _ % 10 == 0:
                loss_array.append(loss.item())
            if _ % 50 == 0:
                print('Epoch: {}, Train Loss: {:.4f}'.format(epoch + 1, loss.item()))
        # 训练集准确率
        train_acc.append(train_cor_count / train_count)
        for img, des, target, idx, mask in valid_dataloader:
            img, mask, idx, target = img.to(device), mask.to(device), idx.to(device), target.to(device)
            output = model(idx, mask, img)
            pred = output.argmax(dim=1)
            valid_cor_count += int(pred.eq(target).sum())
        # 验证集准确率
        valid_acc.append(valid_cor_count / valid_count)
        print('This Epoch is completed.')
        print('Epoch: {}, Train Loss: {:.4f}, Train Accuracy: {:.4f}, Valid Accuracy: {:.4f}'.format(epoch + 1,
                                                                                                     loss.item(),
                                                                                                     train_cor_count / train_count,
                                                                                                     valid_cor_count / valid_count))
    # 绘制准确率的图像
    plt.plot(train_acc, label="Train Accuracy")
    plt.plot(valid_acc, label="Valid Accuracy")
    plt.title(model.__class__.__name__)
    plt.xlabel("Epoch")
    plt.xticks(range(epoch_num), range(1, epoch_num + 1))
    plt.ylabel("Accuracy")
    plt.ylim(ymin=0, ymax=1)
    plt.legend()
    plt.savefig('accuracy_plot.png')
    plt.show()
