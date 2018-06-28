from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.layers import Dense, Activation, Flatten, Convolution2D, Dropout, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.models import Sequential
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from PIL import Image

# 固定随机种子
np.random.seed(42)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf.set_random_seed(42)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

'''
# 修改图片大小
for i in range(10000):
    img_path = '.\猫狗分类器data\train\cat.%d.jpg'%(i)
    im = Image.open(img_path)
    resizedIm = im.resize((224, 224))
    im.save(r'.\猫狗分类器data\\reshape_train\cat.%d.jpg'%(i))
'''

'''
# 数据预处理
# 导入训练好的VGG网络
model = VGG19(weights='imagenet', include_top=False)
train_data_x = np.zeros([20000,7,7,512]).astype("float16")
train_data_y = np.zeros([20000,])
train_data_y[10000:] = 1
# 猫的数据0
for i in range(10000):
    img_path = '.\猫狗分类器data\\reshape_train\cat.%d.jpg'%(i)
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    train_data_x[i, ::] = features
    print(i)

# 狗的数据1
for i in range(10000):
    img_path = '.\猫狗分类器data\\reshape_train\dog.%d.jpg'%(i)
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    train_data_x[i+10000, ::] = features
    print(i+10000)
np.save("./猫狗分类器data/train_data_x.npy",train_data_x)
np.save("./猫狗分类器data/train_data_y.npy",train_data_y)

# 测试数据
test_data_x = np.zeros([5000,7,7,512]).astype("float16")
for i in range(5000):
    img_path = '.\猫狗分类器data\\reshape_test\%d.jpg'%(i)
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    test_data_x[i, ::] = features
    print(i)
np.save("./猫狗分类器data/test_data_x.npy",test_data_x)
'''

# 读取数据
train_data_x = np.load("./猫狗分类器data/train_data_x.npy")
train_data_y = np.load("./猫狗分类器data/train_data_y.npy")
test_data_x = np.load("./猫狗分类器data/test_data_x.npy")
print(train_data_x.shape,train_data_y.shape,test_data_x.shape)  # (20000, 7, 7, 512) (20000,) (5000, 7, 7, 512)

train_data_x = np.reshape(train_data_x,[train_data_x.shape[0], -1])
test_data_x = np.reshape(test_data_x,[test_data_x.shape[0], -1])
print(train_data_x.shape,train_data_y.shape,test_data_x.shape)  # (20000, 25088) (20000,) (5000, 25088)

# 构建模型
model = Sequential()
model.add(Dense(2048,activation="relu",input_shape=(25088,)))
model.add(Dropout(0.5))
model.add(Dense(512,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64))
model.add(Dense(1,activation="sigmoid"))
model.summary()

model.compile(loss='binary_crossentropy',optimizer='adadelta',metrics=['accuracy'])
model.fit(x=train_data_x,y=train_data_y,batch_size=2048,nb_epoch=15,verbose=1)

# 预测答案
test_y = model.predict_classes(test_data_x)
answer = pd.read_csv(open("./猫狗分类器data/sampleSubmission.csv"))
answer["label"] = test_y
answer.to_csv("./猫狗分类器data/submission.csv",index=False)  # 不要保存引索列
