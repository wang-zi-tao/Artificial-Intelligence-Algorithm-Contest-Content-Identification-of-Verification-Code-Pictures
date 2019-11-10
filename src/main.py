import os
import glob
import random
import keras
import numpy as np
import pandas as pd
from keras.preprocessing import image
from keras.callbacks import *
from keras import *
from keras.layers import *
from keras.optimizers import *
from sklearn.model_selection import train_test_split
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import matplotlib.pyplot as plt
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

default_train_times=16
model_file='./model/model.h5'
train_dir = "./验证码图片内容识别竞赛数据/train/"
test_dir = "./验证码图片内容识别竞赛数据/train/"
image_shape=(40,120,3)
test_rate=0.1
batch_size = 128
def int_to_tensor(i,n):
    t=np.zeros((n,))
    t[i]=1.0
    return t
def base64_char_to_int(char):
    c=ord(char)
    if 48<=c<=57:
        return c-48
    if 65<=c<=90:
        return c-65+10
    if 97<=c<=122:
        return c-97+36
        
# 加载训练图片和验证码文字
def load_train_picture_and_label():
    train_label_c0=[]
    train_label_c1=[]
    train_label_c2=[]
    train_label_c3=[]

    test_label_c0=[]
    test_label_c1=[]
    test_label_c2=[]
    test_label_c3=[]

    train_images = []
    test_images = []

    random.seed()
    pd_data = pd.read_csv(test_dir+'train_label.csv')
    for i in range(0, pd_data.shape[0]):
        file = pd_data.iloc[i, 0]
        label = pd_data.iloc[i, 1]
        image_data = image.load_img(train_dir+file)
        image_tensor = image.img_to_array(image_data)
        label_tensor_0=int_to_tensor(base64_char_to_int(label[0]),62)
        label_tensor_1=int_to_tensor(base64_char_to_int(label[1]),62)
        label_tensor_2=int_to_tensor(base64_char_to_int(label[2]),62)
        label_tensor_3=int_to_tensor(base64_char_to_int(label[3]),62)
        if random.random()<test_rate:
            test_images.append(image_tensor)
            test_label_c0.append(label_tensor_0)
            test_label_c1.append(label_tensor_1)
            test_label_c2.append(label_tensor_2)
            test_label_c3.append(label_tensor_3)
        else:
            train_images.append(image_tensor)
            train_label_c0.append(label_tensor_0)
            train_label_c1.append(label_tensor_1)
            train_label_c2.append(label_tensor_2)
            train_label_c3.append(label_tensor_3)

    train_images=np.array(train_images,dtype='float')/255.0
    test_images=np.array(test_images,dtype='float')/255.0

    train_label_c0=np.array(train_label_c0,dtype='float')
    train_label_c1=np.array(train_label_c1,dtype='float')
    train_label_c2=np.array(train_label_c2,dtype='float')
    train_label_c3=np.array(train_label_c3,dtype='float')

    test_label_c0=np.array(test_label_c0,dtype='float')
    test_label_c1=np.array(test_label_c1,dtype='float')
    test_label_c2=np.array(test_label_c2,dtype='float')
    test_label_c3=np.array(test_label_c3,dtype='float')

    train_labels = [train_label_c0,train_label_c1,train_label_c2,train_label_c3]
    test_labels = [test_label_c0,test_label_c1,test_label_c2,test_label_c3]

    return train_images, train_labels,test_images,test_labels

# 加载测试图片和文件名
def load_text_pictrue():
    image_file_list = glob.glob(test_dir+'*.jpg')
    image_name_list = []
    image_data_list = []
    for file in image_file_list:
        image_name_list.append(file)
        image_data = image.load_img(file)
        image_tensor = image.img_to_array(image_data)
        image_data_list.append(image_tensor)
    return image_data_list, image_name_list
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()

def main():
    if True:
        if os.path.exists(model_file):
            model=load_model()
        else:
            model=creat_model(image_shape)
    else:
        model=creat_model(image_shape)
    train_images, train_labels,test_images,test_labels=load_train_picture_and_label()
    train(model,train_images, train_labels,test_images,test_labels)
    store_model(model)

    

# 构建模型
def creat_model(input_shape):
    pic_in = Input(shape=input_shape)
    cnn_features = Conv2D(32, (5,5), activation='relu',kernel_initializer='he_uniform')(pic_in)
    cnn_features = Conv2D(32, (5,5), activation='relu',kernel_initializer='he_uniform')(cnn_features)
    cnn_features = MaxPooling2D((2, 2))(cnn_features)
    #cnn_features = Dropout(0.25)(cnn_features)
    cnn_features = Conv2D(48, (3,3), activation='relu',kernel_initializer='he_uniform')(cnn_features)
    cnn_features = Conv2D(64, (3,3), activation='relu',kernel_initializer='he_uniform')(cnn_features)
    cnn_features = MaxPooling2D((2, 2))(cnn_features)
    cnn_features = Conv2D(96, (3,3), activation='relu',kernel_initializer='he_uniform')(cnn_features)
    cnn_features = Conv2D(128, (3,3), activation='relu',kernel_initializer='he_uniform')(cnn_features)
    cnn_features = MaxPooling2D((2, 2))(cnn_features)
    #cnn_features = Dropout(0.25)(cnn_features)
    #cnn_features = Conv2D(64, (5,5), activation='relu')(cnn_features)
    #cnn_features = Conv2D(64, (5,5), activation='relu')(cnn_features)
    #cnn_features = MaxPooling2D((2, 2))(cnn_features)
    #cnn_features = Dropout(0.25)(cnn_features)
    #cnn_features = Conv2D(64, (5,5), activation='relu')(cnn_features)
    #cnn_features = Conv2D(256, (5,5), activation='relu',padding='same')(cnn_features)
    #cnn_features = MaxPooling2D((2, 2))(cnn_features)
    #cnn_features = Dropout(0.25)(cnn_features)
    cnn_features = Flatten()(cnn_features)
    cnn_features = Dense(1024,activation='relu',kernel_initializer='he_uniform')(cnn_features)
    cnn_features = Dropout(0.25)(cnn_features)
    '''
    cnn_features = Dense(1024,activation='relu',kernel_initializer='he_uniform')(cnn_features)
    cnn_features = Dropout(0.25)(cnn_features)
    cnn_features = Dense(512,activation='relu',kernel_initializer='he_uniform')(cnn_features)
    cnn_features = Dropout(0.25)(cnn_features)
    '''
    # classifier 1
    output_l1 = Dense(128, activation='relu',kernel_initializer='he_uniform')(cnn_features)
    output_l1 = Dropout(0.25)(output_l1)
    output_l1 = Dense(62, activation='softmax',kernel_initializer='he_uniform')(output_l1)
    # classifier 2
    output_l2 = Dense(128, activation='relu',kernel_initializer='he_uniform')(cnn_features)
    output_l2 = Dropout(0.25)(output_l2)
    output_l2 = Dense(62, activation='softmax',kernel_initializer='he_uniform')(output_l2)
    # classifier 3
    output_l3 = Dense(128, activation='relu',kernel_initializer='he_uniform')(cnn_features)
    output_l3 = Dropout(0.25)(output_l3)
    output_l3 = Dense(62, activation='softmax',kernel_initializer='he_uniform')(output_l3)
    # classifier 4
    output_l4 = Dense(128, activation='relu',kernel_initializer='he_uniform')(cnn_features)
    output_l4 = Dropout(0.25)(output_l4)
    output_l4 = Dense(62, activation='softmax',kernel_initializer='he_uniform')(output_l4)


    model = Model(inputs=pic_in, outputs=[output_l1, output_l2, output_l3, output_l4])

    model.compile(
              optimizer='adam',
              loss='categorical_crossentropy',
              loss_weights=[1., 1., 1., 1.], 
              metrics=['accuracy'])

    return model


def train(model,train_images, train_labels,test_images,test_labels,times=default_train_times):
    history=LossHistory()
    callbacks = [
        history
    ]
    model.fit(train_images,train_labels,
          epochs=times,
          #batch_size=batch_size,
          #validation_split=test_rate,
          verbose=1,
          callbacks=callbacks,
          validation_data=(test_images,test_labels)
          #validation_split=0.1
          )
    
    train_predict=model.predict(train_images)
    train_label_tensor=np.array(train_labels)
    print(((train_predict==train_label_tensor).all(axis=(0,2)).sum())/train_label_tensor.shape[1])

    test_predict=model.predict(test_images)
    test_label_tensor=np.array(test_labels)
    print(((test_predict==test_label_tensor).all(axis=(0,2)).sum())/test_label_tensor.shape[1])

    history.loss_plot('epoch')
    return -1


def test():
    # TODO
    return -1


def load_model():
    return keras.models.load_model(model_file)


def store_model(model):
    model.save(model_file)
if __name__ == "__main__":
    main()
