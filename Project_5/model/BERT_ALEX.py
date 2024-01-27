# %% md
# 1.
# 引入必要的Python模块
# %%
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertConfig, BertModel
import torchvision
import torchvision.transforms as transforms

# %% md
# 2.
# Bert与ResNet的混合模型


# %%
class bertAlexModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用Bert与ResNet34的预训练模型
        self.txt_model = BertModel.from_pretrained('bert-base-uncased')
        self.img_model = torchvision.models.alexnet(pretrained=True)
        # 事先定义神经网络层
        self.t_linear = nn.Linear(768, 128)     # 文本的线性层
        self.i_linear = nn.Linear(1000, 128)    # 图片的线性层
        self.img_w = nn.Linear(128, 1)          # 为图片计算权重
        self.txt_w = nn.Linear(128, 1)          # 为文本计算权重
        self.result = nn.Linear(128, 3)         # 三分类全连接层
        self.relu = nn.ReLU()                   # 非线性激活层

    def forward(self, input_ids, attention_mask, image):
        img_out = self.img_model(image)
        img_out = self.i_linear(img_out)
        img_out = self.relu(img_out)
        img_weight = self.img_w(img_out)
        txt_out = self.txt_model(input_ids=input_ids, attention_mask=attention_mask)
        txt_out = txt_out.last_hidden_state[:, 0, :]
        txt_out.view(txt_out.shape[0], -1)
        txt_out = self.t_linear(txt_out)
        txt_out = self.relu(txt_out)
        txt_weight = self.txt_w(txt_out)
        last_out = img_weight * img_out + txt_weight * txt_out
        last_out = self.result(last_out)
        return last_out