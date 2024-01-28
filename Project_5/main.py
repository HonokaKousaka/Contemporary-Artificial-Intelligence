# 引入必要的python库
import numpy as np
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertConfig, BertModel
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import argparse
from model.BERT_RESNET import bertResModel
from model.BERT_ALEX import bertAlexModel
from model.BERT_VGG import bertVGGModel
from model.text import textModel
from model.image import imageModel
from train import train_model

# 调用GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)

# 下载停用词
nltk.download('stopwords')

# 停用词列表
stop_words = set(stopwords.words('english'))


# 将文本转换为Word Embedding，token是文本模型
def word_embed(txt, token):
    result = token.batch_encode_plus(batch_text_or_text_pairs=txt,
                                     truncation=True,
                                     padding='max_length',
                                     max_length=32,
                                     return_tensors='pt')
    input_ids = result['input_ids']
    attention_mask = result['attention_mask']
    return input_ids, attention_mask


# 统一数据格式
# 一个数据包含一张图片，一段描述，一个情感标签。token指的是使用的语言模型
class NewDataset:
    def __init__(self, images, descriptions, tags, token):
        self.images = images
        self.descriptions = descriptions
        self.tags = tags
        self.input_ids, self.attention_masks = word_embed(self.descriptions, token)

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        img = self.images[idx]
        des = self.descriptions[idx]
        tag = self.tags[idx]
        input_id = self.input_ids[idx]
        attention_mask = self.attention_masks[idx]
        return img, des, tag, input_id, attention_mask


def main():
    parser = argparse.ArgumentParser(description="params")
    parser.add_argument('--model', default='bertResModel', help='Choose a model. bertResModel Default.')
    parser.add_argument('--image', action='store_true', help='Image Only Model?')
    parser.add_argument('--text', action='store_true', help='Text Only Model?')
    parser.add_argument('--lr', default=1e-5, type=float, help='Learning rate, 1e-5 Default')
    parser.add_argument('--epoch_num', default=10, type=int, help='Epoch number, 10 Default')
    args = parser.parse_args()
    if args.image:
        model = imageModel().to(device)
    if args.text:
        model = textModel().to(device)
    else:
        if args.model == 'bertResModel':
            model = bertResModel().to(device)
        elif args.model == 'bertAlexModel':
            model = bertAlexModel().to(device)
        elif args.model == 'bertVGGModel':
            model = bertVGGModel().to(device)
        else:
            raise ValueError(f"Unknown Model: {args.model}")
    lr = args.lr
    optimizer = optim.Adam(model.parameters(), lr=lr)
    epoch_num = args.epoch_num

    # 整体准备
    images = []
    descriptions = []
    tags = []
    emo_tag = {"neutral": 0, "negative": 1, "positive": 2}
    token = BertTokenizer.from_pretrained('bert-base-uncased')

    # 读取数据
    train_dataframe = pd.read_csv("./train.txt")
    for i in range(train_dataframe.shape[0]):
        guid = train_dataframe.iloc[i]['guid']
        tag = train_dataframe.iloc[i]['tag']
        img_path = f'./data/{guid}.jpg'
        img = Image.open(img_path).resize((224, 224), Image.Resampling.LANCZOS)
        img = np.asarray(img, dtype='float32')

        txt_path = f'./data/{guid}.txt'
        with open(txt_path, encoding='gb18030') as f:
            des = f.read()

        # 为了符合torch的需求进行转置
        images.append(img.transpose(2, 0, 1))
        descriptions.append(des)
        tags.append(emo_tag[tag])

    # 收录词语
    for i in range(len(descriptions)):
        des = descriptions[i]
        # 空格分词，且去除Hashtag
        word_list = des.replace('#', '').split(' ')
        words_result = []
        for word in word_list:
            # 是一个词，且不是网站，也不是用户名
            if len(word) >= 1 and ('http' not in word) and (word[0] != '@') and word.lower() not in stop_words:
                words_result.append(word)
        descriptions[i] = ' '.join(words_result)

    # 图片与描述一一对应收录
    img_txt_pairs = [(images[i], descriptions[i]) for i in range(len(descriptions))]

    # 划分训练集与验证集
    X_train, X_valid, y_train, y_valid = train_test_split(img_txt_pairs, tags, test_size=0.2, random_state=42,
                                                          shuffle=True)
    image_train, text_train = [X_train[i][0] for i in range(len(X_train))], [X_train[i][1] for i in range(len(X_train))]
    image_valid, text_valid = [X_valid[i][0] for i in range(len(X_valid))], [X_valid[i][1] for i in range(len(X_valid))]

    train_dataset = NewDataset(image_train, text_train, y_train, token)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    valid_dataset = NewDataset(image_valid, text_valid, y_valid, token)
    valid_dataloader = DataLoader(valid_dataset, batch_size=16, shuffle=True)

    # 模型训练
    train_model(model, epoch_num, optimizer, train_dataloader, valid_dataloader, len(X_train), len(X_valid), device)

    # 预测集进行预测
    emo_list = ['neutral', 'negative', 'positive']
    test_df = pd.read_csv('./test_without_label.txt')
    guid_list = test_df['guid'].tolist()
    tag_pre = []
    for idx in guid_list:
        img = Image.open('./data/' + str(idx) + '.jpg')
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        image = np.asarray(img, dtype='float32')
        image = image.transpose(2, 0, 1)
        with open('./data/' + str(idx) + '.txt', encoding='gb18030') as fp:
            description = fp.read()
        input_id, mask = word_embed([description], token)
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
        y_pred = model(input_id.to(device), mask.to(device), torch.Tensor(image).to(device))
        tag_pre.append(emo_list[y_pred[0].argmax(dim=-1).item()])

    result_df = pd.DataFrame({'guid': guid_list, 'tag': tag_pre})
    result_df.to_csv('./result_' + args.model + '.txt', sep=',', index=False)


if __name__ == '__main__':
    main()
