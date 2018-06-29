# -*-coding:utf-8-*-
import keras
import pandas as pd
import numpy as np
from keras.layers import Dense, Activation, Flatten, Convolution2D, Dropout, MaxPooling2D,BatchNormalization
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.models import Sequential
import tensorflow as tf
from keras import backend as K
np.random.seed(42)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf.set_random_seed(42)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

def to_one_hot(y):
    y_temp = np.zeros([y.shape[0], 10])  # 转换为one-hot向量
    for i in range(y.shape[0]):
        y_temp[i, y[i]] = 1
    return y_temp

# 读取数据
train_data = pd.read_csv(open("./Otto_Group_商品识别data/train.csv"))
test_data = pd.read_csv(open("./Otto_Group_商品识别data/test.csv"))
print(train_data.info())
train_y_raw = train_data["target"]
x_label = []
for i in range(1,94):
    x_label.append("feat_%s"%(i))
train_x = np.array(train_data[x_label])
test_x = np.array(test_data[x_label])

# 将train_y中形如Class_1的数据转换成one_hot向量，9维
train_y = np.zeros([len(train_y_raw),9])
for i in range(len(train_y_raw)):
    lable_data = int(train_y_raw[i][-1])  # 取最后一个字就行
    train_y[i,lable_data-1] = 1
print(train_x.shape,train_y.shape,test_x.shape)  # (49502, 93) (49502, 9) (12376, 93)

# 构建模型
model = Sequential()
model.add(Dense(128, input_shape=(93,),activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(9))
model.add(Activation('softmax'))
model.summary()
model.compile(loss='mean_squared_logarithmic_error',optimizer='adadelta',metrics=['accuracy'])
model.fit(x=train_x,y=train_y,batch_size=2048,nb_epoch=200,verbose=1)

# 预测答案
test_y = model.predict(test_x)
print(test_y.shape)
answer = pd.read_csv(open("./Otto_Group_商品识别data/sampleSubmission.csv"))
class_list = ["Class_1","Class_2","Class_3","Class_4","Class_5","Class_6","Class_7","Class_8","Class_9"]
answer[class_list] = answer[class_list].astype(float)

# 将答案放进去
j = 0
for class_name in class_list:
    answer[class_name] = test_y[:, j]
    j += 1
answer.to_csv("./Otto_Group_商品识别data/submission.csv",index=False)  # 不要保存引索列
