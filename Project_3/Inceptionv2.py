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
parser = argparse.ArgumentParser(description="The parameters of Inceptionv2")

parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--dropout', type=float, default=0.4)
args = parser.parse_args()
lr = args.lr
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
搭建Inceptionv2
"""
# 初始化模块与深度级联
def inception_module(x, filters):
  conv1_1_1_1 = layers.Conv2D(filters[0], (1, 1), padding='same', activation='relu')(x)
  conv1_1_1_2 = layers.Conv2D(filters[1], (1, 1), padding='same', activation='relu')(x)
  conv3_3_1 = layers.Conv2D(filters[2], (3, 3), padding='same', activation='relu')(conv1_1_1_2)
  conv1_1_1_3 = layers.Conv2D(filters[3], (1, 1), padding='same', activation='relu')(x)
  conv5_5_1 = layers.Conv2D(filters[4], (5, 5), padding='same', activation='relu')(conv1_1_1_3)
  maxpool = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
  conv1_1_1_4 = layers.Conv2D(filters[5], (1, 1), padding='same', activation='relu')(maxpool)

  conv1_1_1_1 = layers.BatchNormalization()(conv1_1_1_1)
  conv3_3_1 = layers.BatchNormalization()(conv3_3_1)
  conv5_5_1 = layers.BatchNormalization()(conv5_5_1)
  conv1_1_1_4 = layers.BatchNormalization()(conv1_1_1_4)

  # 深度级联
  inception = tf.concat([conv1_1_1_1, conv3_3_1, conv5_5_1, conv1_1_1_4], axis=-1)

  return inception

# 搭建神经网络
def Inceptionv2_model(input_shape=(32, 32, 1), num_classes=10):
  input_tensor = layers.Input(shape=input_shape)
  # x = layers.Conv2D(64, (7, 7), padding='same', activation='relu', strides=(2, 2))(input_tensor)
  # x = layers.MaxPooling2D((3, 3), padding='same', strides=(2, 2))(x)
  # x = layers.Lambda(lambda x: tf.nn.local_response_normalization(x))(x)
  x = layers.Conv2D(64, (1, 1), padding='same', activation='relu', strides=(1, 1))(input_tensor)
  x = layers.Conv2D(192, (3, 3), padding='same', activation='relu', strides=(1, 1))(x)
  x = layers.Lambda(lambda x: tf.nn.local_response_normalization(x))(x)
  x = layers.MaxPooling2D((3, 3), padding='same', strides=(2, 2))(x)

  x = inception_module(x, [64, 96, 128, 16, 32, 32])
  x = inception_module(x, [128, 128, 192, 32, 96, 64])
  x = layers.MaxPooling2D((3, 3), padding='same', strides=(2, 2))(x)
  x = inception_module(x, [192, 96, 208, 16, 48, 64])
  # x = inception_module(x, [160, 112, 224, 24, 64, 64])
  # x = inception_module(x, [128, 128, 256, 24, 64, 64])
  # x = inception_module(x, [112, 144, 288, 32, 64, 64])
  x = inception_module(x, [256, 160, 320, 32, 128, 128])

  x = layers.MaxPooling2D((3, 3), padding='same', strides=(2, 2))(x)
  x = inception_module(x, [256, 160, 320, 32, 128, 128])
  x = inception_module(x, [384, 192, 384, 48, 128, 128])
  x = layers.AveragePooling2D((2,2), padding='valid', strides=(1, 1))(x)
  x = layers.Dropout(do)(x)
  x = layers.Flatten()(x)
  x = layers.Dense(512, activation='relu')(x)
  output = layers.Dense(num_classes, activation='softmax')(x)

  model = models.Model(inputs=input_tensor, outputs=output)
  return model
model = Inceptionv2_model()

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