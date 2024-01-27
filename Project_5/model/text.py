# 只读取文本
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertConfig, BertModel

class textModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.txt_model = BertModel.from_pretrained('bert-base-uncased')
        self.t_linear = nn.Linear(768, 256)
        self.fc = nn.Linear(256, 3)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask, image):
        txt_out = self.txt_model(input_ids=input_ids, attention_mask=attention_mask)
        txt_out = txt_out.last_hidden_state[:,0,:]
        txt_out.view(txt_out.shape[0],-1)
        txt_out = self.t_linear(txt_out)
        txt_out = self.relu(txt_out)
        last_out = self.fc(txt_out)
        return last_out