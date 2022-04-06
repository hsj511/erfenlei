# 导入头文件
import os, cv2, time
import numpy as np
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow import keras
import tensorflow as tf

# 设置相关参数
nRow, nCol, nChannel = 224, 224, 3 # 样本归一化尺寸
fTrainSplit = 0.6  # 训练集在训练数据中占的比例
fVerifySplit = 0.2 # 验证集在训练数据中占的比例

# 模型训练配置
fLr = 0.0001
nEpochs = 20
opt=Adam(lr = fLr, decay = fLr / (nEpochs / 0.5))

# 1. 加载训练数据
sPathTrainingData = r'D:\erfenlei\bmp\train'   # 指定训练数据集路径
sPathTestingData = r'D:\erfenlei\bmp\test'     # 指定测试数据集路径

slSubFolderTrainingData = os.listdir(sPathTrainingData) # 训练数据集子文件夹名列表，正样本文件夹名为“1”，负样本文件夹名为“0”
slSubFolderTestingData = os.listdir(sPathTestingData)   # 测试数据集子文件夹名列表，正样本文件夹名为“1”，负样本文件夹名为“0”

ilTrainingImage = []    # 建立列表，用于保存训练样本图像
nlTrainingLabel = []    # 建立列表，用于保存训练样本标签
# 制作训练样本和标签数据集
for sSubFolderTrain in slSubFolderTrainingData: # 遍历训练数据集子文件夹（“0”和“1”）
    sPathSubFolderTrainingData = sPathTrainingData + "\\" + sSubFolderTrain # 获取每个子文件夹的路径
    nLabelTrainingData = int(sSubFolderTrain) # 直接拿文件夹名作为标签（正样本标签1，负样本标签0）
    slTrainingImageName = os.listdir(sPathSubFolderTrainingData) # 获取正负训练样本的文件名列表
    for sImg in slTrainingImageName: # 遍历每个样本
        iImage = cv2.imread(sPathSubFolderTrainingData + "\\" + sImg) # 读取图像
        iImage = cv2.resize(iImage, (nRow, nCol))    # 训练图像尺寸归一化
        iImage = iImage / 255.                       # 灰度值归一化
        ilTrainingImage.append(iImage)               # 图像塞入训练样本列表
        nlTrainingLabel.append(nLabelTrainingData)   # 标签塞入训练样本标签列表

# 这里把测试样本也都塞入训练样本列表，和训练样本标签列表，到下一步在划分：训练集，验证集，测试集
for sSubFolderTest in slSubFolderTestingData:
    sPathSubFolderTestingData = sPathTestingData + "\\" + sSubFolderTest
    nLabelTestingData = int(sSubFolderTest)
    slTestingImageName = os.listdir(sPathSubFolderTestingData)
    for sImg in slTestingImageName:
        iImage = cv2.imread(sPathSubFolderTestingData + "\\" + sImg)
        iImage = cv2.resize(iImage, (nRow, nCol))
        iImage = iImage / 255.
        ilTrainingImage.append(iImage)
        nlTrainingLabel.append(nLabelTestingData)

state = np.random.get_state() # 把训练数据打乱顺序
np.random.shuffle(ilTrainingImage)
np.random.set_state(state)
np.random.shuffle(nlTrainingLabel)

nNumberTrain = int(len(nlTrainingLabel) * fTrainSplit)                     # 样本中用于训练的样本个数 fTrainSplit=0.6
nNumberVerify = int(len(nlTrainingLabel) * fVerifySplit)                   # 样本中用于验证的样本个数 fVerifySplit=0.2
nNumberTest = int(len(nlTrainingLabel) * (1 - fTrainSplit - fVerifySplit)) # 样本中用于测试的样本个数

xTrain = np.array(ilTrainingImage[0:nNumberTrain], np.float32)   # 训练样本中用于训练的数据
yTrain = np.array(nlTrainingLabel[0:nNumberTrain], np.int)
yTrain = keras.utils.to_categorical(yTrain, 2)

xVerify = np.array(ilTrainingImage[nNumberTrain:nNumberTrain+nNumberVerify], np.float32) # 训练样本中用于验证的数据
yVerify = np.array(nlTrainingLabel[nNumberTrain:nNumberTrain+nNumberVerify], np.int)
yVerify = keras.utils.to_categorical(yVerify, 2)

xTest = np.array(ilTrainingImage[-nNumberTest:], np.float32) # 训练样本中用于测试的数据（不参与训练）
yTest = np.array(nlTrainingLabel[-nNumberTest:], np.int)
yTest = keras.utils.to_categorical(yTest, 2)

del ilTrainingImage # 防止内存溢出
del nlTrainingLabel

# 2. 创建模型并训练
covn_base = tf.keras.applications.InceptionV3(weights='imagenet',include_top=False,input_shape=(224,224,3)) # finetune权重'imagenet'，不是初始化参数
covn_base.trainable = True

#冻结前面的层，训练最后20层（当然冻结多少层自己定）
for layers in covn_base.layers[:-20]:
    layers.trainable = False

#构建模型
model = tf.keras.Sequential()
model.add(covn_base)
model.add(tf.keras.layers.GlobalAveragePooling2D())       # 加入全局平均池化层
model.add(tf.keras.layers.Dense(2, activation='softmax')) # 加入输出层(2分类)
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['acc']) # 选择优化器，损失函数，评测指标
history = model.fit(xTrain, yTrain, batch_size=20, epochs=nEpochs, validation_data=(xVerify, yVerify))
del xTrain # 防止内存溢出
del yTrain
del xVerify
del yVerify

model.save_weights('weights.ckpt')
del model

# 3. 对测试样本进行处理
model = tf.keras.Sequential()
model.add(covn_base)
model.add(tf.keras.layers.GlobalAveragePooling2D()) #加入全局平均池化层
model.add(tf.keras.layers.Dense(2, activation='softmax')) #加入输出层(2分类)
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['acc']) # 选择优化器，损失函数，评测指标
model.load_weights('weights.ckpt')

loss, accuracy = model.evaluate(xTest, yTest)
import matplotlib.pyplot as plt
# 记录训练集和验证集的准确率和损失值
history_dict = history.history
train_loss = history_dict["loss"] #训练集损失值
train_accuracy = history_dict["accuracy"] #训练集准确率
val_loss = history_dict["val_loss"] #验证集损失值
val_accuracy = history_dict["val_accuracy"] #验证集准确率
plt.figure()
plt.plot(range(nEpochs), train_loss, label='train_loss')
plt.plot(range(nEpochs), val_loss, label='val_loss')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.figure()
plt.plot(range(nEpochs), train_accuracy, label='train_accuracy')
plt.plot(range(nEpochs), val_accuracy, label='val_accuracy')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()
