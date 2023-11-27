import argparse
import time as time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

"""
命令行解析器
"""
parser = argparse.ArgumentParser(description="The parameters of LeNet")

parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--dropout', type=float, default=0.0)
args = parser.parse_args()
lr = args.lr
# 实际上没有用到Dropout
do = args.dropout
"""
读取MNIST数据集
"""
# These variables are all in type of numpy.
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# print(train_images.shape)
# print(train_labels.shape)
# print(test_images.shape)
# print(test_labels.shape)

"""
为28*28的图片增加Padding，使得规模变为32*32
同时为标签作One-hot编码，适配categorical_crossentropy的损失函数
"""
train_images_32 = np.zeros((60000, 32, 32), dtype=train_images.dtype)
test_images_32 = np.zeros((10000, 32, 32), dtype=test_images.dtype)

start_row = (32 - 28) // 2
start_col = (32 - 28) // 2
for i in range(60000):
  train_images_32[i][start_row:start_row+28, start_col:start_col+28] = train_images[i]
for i in range(10000):
  test_images_32[i][start_row:start_row+28, start_col:start_col+28] = test_images[i]

train_images_32 = train_images_32.reshape((60000, 32, 32, 1)).astype('float32') / 255
test_images_32 = test_images_32.reshape((10000, 32, 32, 1)).astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# print(train_images_32.shape)
# print(test_images_32.shape)
# print(train_labels.shape)
# print(test_labels.shape)

"""
搭建LeNet
"""
model = models.Sequential()
model.add(layers.Conv2D(6, (5, 5), activation='relu', input_shape=(32, 32, 1)))
model.add(layers.AveragePooling2D((2,2)))
model.add(layers.Conv2D(16, (5, 5), activation='relu'))
model.add(layers.AveragePooling2D((2, 2)))
model.add(layers.Conv2D(120, (5, 5), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(84, activation='relu'))
model.add(layers.Dense(10, activation='sigmoid'))

"""
编译模型
"""
custom_optimizer = optimizers.Adam(learning_rate=lr)
model.compile(optimizer=custom_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

"""
训练模型
"""
start_time = time.time()
model.fit(train_images_32, train_labels, epochs=5, batch_size=64, validation_split=0.2)
end_time = time.time()
print("Running Time:", end_time-start_time, "seconds")

"""
模型用于预测
"""
test_loss, test_acc = model.evaluate(test_images_32, test_labels)
print(f'Test accuracy: {test_acc}')
