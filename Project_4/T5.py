# %% md
# Text Summarization
## —— Based on T5 (Text-To-Text Transfer Transformer)
# %% md
# Reference github repository: https://github.com/abhimishra91/transformers-tutorials/blob/master/transformers_summarization_wandb.ipynb
# %% md
### 1. 调取必要模块
# %%
# 基础模块
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rouge import Rouge
from sklearn.model_selection import KFold
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate import meteor_score
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

# 基于Transformers架构的T5框架
from transformers import T5Tokenizer, T5ForConditionalGeneration

# %%
# 这段代码的目的是在AutoDL云上服务器中开启加速
# 从而更快地获取来自HuggingFace的内容
# 选自AutoDL的帮助文档
# 个人建议在AutoDL的环境下运行，或直接不运行这段代码。否则可能出现错误。

"""
如果在载入模块时连接时间过长，可以启用这段代码
建议在AutoDL的环境下运行，或自行配置合适的环境
否则可能出现错误！
"""
# import subprocess
# import os
#
# result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
# output = result.stdout
# for line in output.splitlines():
#     if '=' in line:
#         var, value = line.split('=', 1)
#         os.environ[var] = value
# %%
# 调用GPU
# 本人租用的服务器采用了具有2块GPU的RTX 4090
from torch import cuda

device = "cuda" if cuda.is_available() else "cpu"

print(device)
# %% md
### 2. 自定义数据集
# 基于train.csv设计的自定义数据集，为后续神经网络的微调做准备。

# %%
class MedicalDataset(Dataset):

    def __init__(self, dataframe, tokenizer, source_len, summ_len):
        # tokenizer: 用于分词
        self.tokenizer = tokenizer
        # data: 存储数据
        self.data = dataframe
        # description的内容的长度
        self.source_len = source_len
        # diagnosis的内容的长度
        self.summ_len = summ_len
        # csv文件中description列的内容
        self.description = self.data.description
        # csv文件中diagnosis列的内容
        self.diagnosis = self.data.diagnosis

    # 返回数据的数量
    def __len__(self):
        return len(self.description)

    def __getitem__(self, index):
        # 如果文本中有多个连续空格
        # 则合并为一个空格
        diagnosis = str(self.diagnosis[index])
        diagnosis = ' '.join(diagnosis.split())
        description = str(self.description[index])
        description = ' '.join(description.split())
        # batch_encode_plus将文本转为模型可以处理的编码
        source = self.tokenizer.batch_encode_plus([description],
                                                  max_length=self.source_len,  # 指定编码后文本最大长度
                                                  padding='max_length',  # 填充使得每个文本长度相同
                                                  return_tensors='pt',  # 返回Pytorch的张量
                                                  truncation=True)  # 如果超过最大长度，需要截断
        target = self.tokenizer.batch_encode_plus([diagnosis],
                                                  max_length=self.summ_len,  # 指定编码后文本最大长度
                                                  padding='max_length',  # 填充使得每个文本长度相同
                                                  return_tensors='pt',  # 返回Pytorch的张量
                                                  truncation=True)  # 如果超过最大长度，需要截断

        # Pytorch中的squeeze函数移除大小为1的维度
        # 张量形状更紧凑
        # input_ids返回了编码后的文本的token IDs
        # attention_mask二进制掩码，表明哪些位置是填充，注意力机制不应放在填充位置
        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long)
        }


# %% md
### 3. 编写训练函数
# 训练集会经过训练函数，从而达成模型的微调。
# %%
loss_array = []


def train(epoch, tokenizer, model, device, loader, optimizer):
    model.train()
    for _, data in enumerate(loader):
        y = data['target_ids'].to(device, dtype=torch.long)
        # 获取 y 的每一行，且不要最后一个标记
        # 用contiguous使张量连续
        y_ids = y[:, :-1].contiguous()
        # detach让labels不用参与梯度计算
        labels = y[:, 1:].clone().detach()
        # 填充位置对应的损失值设为-100
        # -100是Pytorch中的一个特殊值，计算损失时会忽略这些位置
        labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data['source_ids'].to(device, dtype=torch.long)
        mask = data['source_mask'].to(device, dtype=torch.long)

        outputs = model(input_ids=ids,
                        attention_mask=mask,
                        decoder_input_ids=y_ids,
                        labels=labels)
        loss = outputs[0]

        if _ % 10 == 0:
            loss_array.append(loss.item())
        if _ % 500 == 0:
            print(f'Epoch: {epoch + 1}, Loss: {loss.item()}')

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# %% md
### 4. 编写验证函数
# %%
def validate(tokenizer, model, device, loader):
    model.eval()
    predictions = []
    actual = []
    with torch.no_grad():
        for _, data in enumerate(loader):
            y = data['target_ids'].to(device, dtype=torch.long)
            ids = data['source_ids'].to(device, dtype=torch.long)
            mask = data['source_mask'].to(device, dtype=torch.long)

            generated_ids = model.generate(input_ids=ids,
                                           attention_mask=mask,
                                           max_length=150,
                                           # Beam Search的参数数量
                                           num_beams=2,
                                           # 重复惩罚
                                           repetition_penalty=2.5,
                                           # 长度惩罚
                                           length_penalty=1.0,
                                           # 要求在遇到End of Sequence时停止
                                           # 若设置为False，会生成直到最大长度
                                           early_stopping=True)
            preds = [tokenizer.decode(g,
                                      skip_special_tokens=True,
                                      clean_up_tokenization_spaces=True) for g in generated_ids]

            target = [tokenizer.decode(t,
                                       skip_special_tokens=True,
                                       clean_up_tokenization_spaces=True) for t in y]

            if _ % 100 == 0:
                print(f"Completed {_}")

            predictions.extend(preds)
            actual.extend(target)
        return predictions, actual


# %% md
### 5. 五折交叉验证训练模型
# %%
# 设定参数
TRAIN_BATCH_SIZE = 2  # 训练的批次大小
VALID_BATCH_SIZE = 2  # 测试的批次大小
TRAIN_EPOCHS = 3  # 训练的轮次数量
LEARNING_RATE = 1e-4  # 学习率
SEED = 42  # 随机种子
MAX_LEN = 512  # source_len
SUMMARY_LEN = 150  # summ_len
# %%
# 设置随机种子
torch.manual_seed(SEED)
np.random.seed(SEED)

# 可复现性
torch.backends.cudnn.deterministic = True
# %%
# Tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-base")
# %%
# 导入数据
df = pd.read_csv("train.csv")
df = df[['description', 'diagnosis']]
print(df.head())
# %%
des_len_list = []
diag_len_list = []
des_list = df['description'].tolist()
diag_list = df['diagnosis'].tolist()
for i in range(len(des_list)):
    des_len_list.append(len(des_list[i].split()))
for i in range(len(diag_list)):
    diag_len_list.append(len(diag_list[i].split()))

des_len_list = np.array(des_len_list)
diag_len_list = np.array(diag_len_list)
# %%
# 使用 numpy.unique 函数统计每个值的出现次数
unique_des, counts_des = np.unique(des_len_list, return_counts=True)
unique_diag, counts_diag = np.unique(diag_len_list, return_counts=True)
# %%
import matplotlib.pyplot as plt

plt.bar(unique_des, counts_des)
plt.show()
# %%
plt.bar(unique_diag, counts_diag)
plt.show()
# %%
# 定义参数
# num_workers = 0 表示在主进程中加载数据
train_params = {
    'batch_size': TRAIN_BATCH_SIZE,
    'shuffle': True,
    'num_workers': 0
}

val_params = {
    'batch_size': VALID_BATCH_SIZE,
    'shuffle': False,
    'num_workers': 0
}
# %%
# 正式开始训练
print("开始模型微调")

num = 1
kfolds = KFold(n_splits=5, shuffle=True, random_state=42)
for train_index, test_index in kfolds.split(df):
    des_trainset = df['description'][train_index]
    dia_trainset = df['diagnosis'][train_index]
    des_valset = df['description'][test_index]
    dia_valset = df['diagnosis'][test_index]
    train_dataset = pd.concat([des_trainset, dia_trainset], axis=1).reset_index(drop=True)
    val_dataset = pd.concat([des_valset, dia_valset], axis=1).reset_index(drop=True)

    training_set = MedicalDataset(train_dataset, tokenizer, MAX_LEN, SUMMARY_LEN)
    val_set = MedicalDataset(val_dataset, tokenizer, MAX_LEN, SUMMARY_LEN)

    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)

    # 使用预训练的T5-base模型
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    model = model.to(device)

    # 定义优化器，使用Adam
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    for epoch in range(TRAIN_EPOCHS):
        train(epoch, tokenizer, model, device, training_loader, optimizer)

    print("在验证集上测试")

    predictions, actual = validate(tokenizer, model, device, val_loader)
    final_df = pd.DataFrame({'Generated Text': predictions, 'Actual Text': actual})
    final_df.to_csv('predictions' + str(num) + '.csv')
    num = num + 1

    print("输出文件已保存")
# %% md
### 6. 测试在验证集上的表现
# %%
for i in range(1, 6):
    df = pd.read_csv("predictions" + str(i) + ".csv")
    rouge1_p = 0
    rouge1_r = 0
    rouge1_f = 0
    rouge2_p = 0
    rouge2_r = 0
    rouge2_f = 0
    rougel_p = 0
    rougel_r = 0
    rougel_f = 0
    bleu_average = 0
    meteor_average = 0
    rouge = Rouge()
    for i in range(3600):
        answer = df["Actual Text"][i]
        summary = df["Generated Text"][i]
        scores = rouge.get_scores(summary, answer)
        rouge1_p += scores[0]["rouge-1"]["p"] / 3600
        rouge1_r += scores[0]["rouge-1"]["r"] / 3600
        rouge1_f += scores[0]["rouge-1"]["f"] / 3600
        rouge2_p += scores[0]["rouge-2"]["p"] / 3600
        rouge2_r += scores[0]["rouge-2"]["r"] / 3600
        rouge2_f += scores[0]["rouge-2"]["f"] / 3600
        rougel_p += scores[0]["rouge-l"]["p"] / 3600
        rougel_r += scores[0]["rouge-l"]["r"] / 3600
        rougel_f += scores[0]["rouge-l"]["f"] / 3600
        answer_nltk = [answer]
        bleu_average += sentence_bleu(answer_nltk, summary) / 3600
        summary_tokens = summary.split()
        answer_nltk_2 = [reference.split() for reference in answer_nltk]
        meteor_average += meteor_score.meteor_score(answer_nltk_2, summary_tokens) / 3600

    print("The precision of Rouge 1 is:", rouge1_p)
    print("The recall of Rouge 1 is:", rouge1_r)
    print("The F1 of Rouge 1 is:", rouge1_f)
    print("The precision of Rouge 2 is:", rouge2_p)
    print("The recall of Rouge 2 is:", rouge2_r)
    print("The F1 of Rouge 2 is:", rouge2_f)
    print("The precision of Rouge l is:", rougel_p)
    print("The recall of Rouge l is:", rougel_r)
    print("The F1 of Rouge l is:", rougel_f)
    print("BLEU Score:", bleu_average)
    print("METEOR Score:", meteor_average)
    print("----------------------------------------")
# %% md
### 7. 使用全部训练集训练模型
# %%
# 导入数据
df = pd.read_csv("train.csv")
df = df[['description', 'diagnosis']]
print(df.head())

# 创建符合格式预期的数据集
training_set = MedicalDataset(df, tokenizer, MAX_LEN, SUMMARY_LEN)
training_loader = DataLoader(training_set, **train_params)

# 使用预训练的T5-base模型
model = T5ForConditionalGeneration.from_pretrained("t5-base")
model = model.to(device)

# 定义优化器，使用Adam
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
# %%
# 正式开始训练
print("开始模型微调")

for epoch in range(TRAIN_EPOCHS):
    train(epoch, tokenizer, model, device, training_loader, optimizer)
# %%
# 绘制损失函数曲线图
import matplotlib.pyplot as plt

# 创建 x 轴的坐标，这里使用列表的索引作为 x 轴坐标
x = list(range(1, len(loss_array) + 1))

plt.plot(x, loss_array, color='b', label='Loss')

plt.title('Loss Through the Training Process')
plt.ylabel('Loss')

plt.legend()
plt.show()
# %%
df_test = pd.read_csv("test.csv")
val_dataset = df_test[['description', 'diagnosis']]
print(df_test.head())

val_set = MedicalDataset(val_dataset, tokenizer, MAX_LEN, SUMMARY_LEN)
val_loader = DataLoader(val_set, **val_params)
# %%
print("在测试集上测试")

predictions, actual = validate(tokenizer, model, device, val_loader)
final_df = pd.DataFrame({'Generated Text': predictions, 'Actual Text': actual})
final_df.to_csv('predictions.csv')

print("输出文件已保存")
# %% md
### 8. 评价在测试集上的表现
# %%
df = pd.read_csv("predictions.csv")
rouge1_p = 0
rouge1_r = 0
rouge1_f = 0
rouge2_p = 0
rouge2_r = 0
rouge2_f = 0
rougel_p = 0
rougel_r = 0
rougel_f = 0
bleu_average = 0
meteor_average = 0
rouge = Rouge()
for i in range(2000):
    answer = df["Actual Text"][i]
    summary = df["Generated Text"][i]
    scores = rouge.get_scores(summary, answer)
    rouge1_p += scores[0]["rouge-1"]["p"] / 2000
    rouge1_r += scores[0]["rouge-1"]["r"] / 2000
    rouge1_f += scores[0]["rouge-1"]["f"] / 2000
    rouge2_p += scores[0]["rouge-2"]["p"] / 2000
    rouge2_r += scores[0]["rouge-2"]["r"] / 2000
    rouge2_f += scores[0]["rouge-2"]["f"] / 2000
    rougel_p += scores[0]["rouge-l"]["p"] / 2000
    rougel_r += scores[0]["rouge-l"]["r"] / 2000
    rougel_f += scores[0]["rouge-l"]["f"] / 2000
    answer_nltk = [answer]
    bleu_average += sentence_bleu(answer_nltk, summary) / 2000
    summary_tokens = summary.split()
    answer_nltk_2 = [reference.split() for reference in answer_nltk]
    meteor_average += meteor_score.meteor_score(answer_nltk_2, summary_tokens) / 2000

print("The precision of Rouge 1 is:", rouge1_p)
print("The recall of Rouge 1 is:", rouge1_r)
print("The F1 of Rouge 1 is:", rouge1_f)
print("The precision of Rouge 2 is:", rouge2_p)
print("The recall of Rouge 2 is:", rouge2_r)
print("The F1 of Rouge 2 is:", rouge2_f)
print("The precision of Rouge l is:", rougel_p)
print("The recall of Rouge l is:", rougel_r)
print("The F1 of Rouge l is:", rougel_f)
print("BLEU Score:", bleu_average)
print("METEOR Score:", meteor_average)