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
        # 使用Bert与AlexNet的预训练模型
        self.textual_model = BertModel.from_pretrained('bert-base-uncased')
        self.visual_model = torchvision.models.alexnet(pretrained=True)
        # 事先定义神经网络层
        self.textual_linear = nn.Linear(768, 128)   # 文本的线性层
        self.visual_linear = nn.Linear(1000, 128)   # 图片的线性层
        self.image_weight = nn.Linear(128, 1)       # 为图片计算权重
        self.text_weight = nn.Linear(128, 1)        # 为文本计算权重
        self.result = nn.Linear(128, 3)             # 三分类全连接层
        self.relu = nn.ReLU()                       # 非线性激活层

    def forward(self, input_ids, attention_mask, image):
        visual_output = self.visual_model(image)
        visual_output = self.visual_linear(visual_output)
        visual_output = self.relu(visual_output)
        image_weight = self.image_weight(visual_output)

        textual_output = self.textual_model(input_ids=input_ids, attention_mask=attention_mask)
        textual_output = textual_output.last_hidden_state[:, 0, :]
        textual_output.view(textual_output.shape[0], -1)
        textual_output = self.textual_linear(textual_output)
        textual_output = self.relu(textual_output)
        text_weight = self.text_weight(textual_output)

        final_output = image_weight * visual_output + text_weight * textual_output
        final_output = self.result(final_output)
        return final_output