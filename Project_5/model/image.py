# 只读取图片
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

class imageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_model = torchvision.models.resnet34(pretrained=True)
        self.i_linear = nn.Linear(1000, 256)
        self.fc = nn.Linear(256, 3)
        self.relu = nn.ReLU()
    
    def forward(self, input_ids, attention_mask, image):
        img_out = self.img_model(image)
        img_out = self.i_linear(img_out)
        img_out = self.relu(img_out)
        last_out = self.fc(img_out)
        return last_out