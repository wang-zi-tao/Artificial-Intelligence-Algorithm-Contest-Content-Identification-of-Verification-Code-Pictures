import os
import glob
import random
import keras
import json
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
'''
default_train_times=16
model_file='./model/model.h5'
predict_result_file='./predict_result.csv'
train_dir = "./验证码图片内容识别竞赛数据/train/"
test_dir = "./验证码图片内容识别竞赛数据/test/"

test_rate=0.1
batch_size = 128
'''
image_shape=(40,120,3)
def compare(predict,label):
    n=predict[0].shape[0]
    '''
    c=0
    for i in range(0,n):
        for j in range(0,4):
            l=np.argmax(predict[j,i])
            if(label[j,l]!=1):
                break
        else:
            c+=1
    return c/n
    '''
    label_argmax=np.array(label).argmax(axis=2)
    predict_argmax=np.array(predict).argmax(axis=2)
    
    #print(label_argmax)
    #print(predict_argmax)
    return (label_argmax==predict_argmax).all(axis=(0,)).sum()/n

def int_to_tensor(i,n):
    t=np.zeros((n,))
    t[i]=1.0
    return t
def int_to_base64_char(i):
    if 0<=i<=9:
        return chr(i+48)
    if 10<=i<=35:
        return chr(i+55)
    if 36<=i<=61:
        return chr(i+61)
def base64_char_to_int(char):
    c=ord(char)
    if 48<=c<=57:
        return c-48
    if 65<=c<=90:
        return c-65+10
    if 97<=c<=122:
        return c-97+36
        
# 加载训练图片和验证码文字
def load_train_picture_and_label(config):

    train_label_c0=[]
    train_label_c1=[]
    train_label_c2=[]
    train_label_c3=[]

    train_images = []

    image_dir=config['源图路径']

    random.seed()
    pd_data = pd.read_csv(image_dir+'train_label.csv')
    for i in range(0, pd_data.shape[0]):
        file = pd_data.iloc[i, 0]
        label = pd_data.iloc[i, 1]
        image_data = image.load_img(image_dir+file)
        image_tensor = image.img_to_array(image_data)
        label_tensor_0=int_to_tensor(base64_char_to_int(label[0]),62)
        label_tensor_1=int_to_tensor(base64_char_to_int(label[1]),62)
        label_tensor_2=int_to_tensor(base64_char_to_int(label[2]),62)
        label_tensor_3=int_to_tensor(base64_char_to_int(label[3]),62)
        train_images.append(image_tensor)
        train_label_c0.append(label_tensor_0)
        train_label_c1.append(label_tensor_1)
        train_label_c2.append(label_tensor_2)
        train_label_c3.append(label_tensor_3)

    train_images=np.array(train_images,dtype='float')/255.0

    train_label_c0=np.array(train_label_c0,dtype='float')
    train_label_c1=np.array(train_label_c1,dtype='float')
    train_label_c2=np.array(train_label_c2,dtype='float')
    train_label_c3=np.array(train_label_c3,dtype='float')

    train_labels = [train_label_c0,train_label_c1,train_label_c2,train_label_c3]

    return train_images, train_labels

# 加载测试图片和文件名
def load_predict_pictrue(config):
    image_name_list = []
    image_data_list = []
    image_dir=config['源图路径']
    for i in range(1,5001):
        file=str(i)+'.jpg'
        image_name_list.append(file)
        image_data = image.load_img(image_dir+file)
        image_tensor = image.img_to_array(image_data)
        image_data_list.append(image_tensor)
    image_data_list=np.array(image_data_list,dtype='float')/255.0
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
    config=json.loads(open('./config.json',encoding='utf-8').read())
    if config['模型']['放弃']:
        model=creat_model(image_shape)
    else:
        if os.path.exists(config['模型']['文件路径']):
            model=load_model(config['模型'])
        else:
            model=creat_model(image_shape)
    if config['训练']['启用']:
        train_config=config['训练']
        train_images, train_labels=load_train_picture_and_label(train_config['训练集'])
        if train_config['循环训练']:
            while True:
                train(model,train_images, train_labels,train_config)
                store_model(model,config['模型'])
        else:
            train(model,train_images, train_labels,train_config)
            store_model(model,config['模型'])
    if config['预测']['启用']:
        predict_config=config['预测']
        predict_image,predict_file_name=load_predict_pictrue(predict_config['预测集'])
        predict_label=predict(model,predict_image)
        save_predict_result(predict_file_name,predict_label,predict_config)



# 构建模型
def creat_model(input_shape):
    cnn_features = pic_in = Input(shape=input_shape)
    cnn_features = GaussianDropout(0.4)(cnn_features)

    cnn_features = Conv2D(32, (5,5), activation='relu',kernel_initializer='he_uniform')(cnn_features)
    cnn_features = Conv2D(32, (5,5), activation='relu',kernel_initializer='he_uniform')(cnn_features)
    cnn_features = MaxPooling2D((2, 2))(cnn_features)
    cnn_features = Dropout(0.25)(cnn_features)
    cnn_features = Conv2D(48, (3,3), activation='relu',kernel_initializer='he_uniform')(cnn_features)
    cnn_features = Conv2D(48, (3,3), activation='relu',kernel_initializer='he_uniform')(cnn_features)
    cnn_features = MaxPooling2D((2, 2))(cnn_features)
    cnn_features = Dropout(0.25)(cnn_features)
    cnn_features = Conv2D(96, (3,3), activation='relu',kernel_initializer='he_uniform')(cnn_features)
    cnn_features = Conv2D(128, (3,3), activation='relu',kernel_initializer='he_uniform')(cnn_features)
    cnn_features = MaxPooling2D((2, 2))(cnn_features)
    cnn_features = Dropout(0.25)(cnn_features)
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
    output_l1 = Dense(128, activation='relu',kernel_initializer='he_uniform',kernel_regularizer=regularizers.l2(0.01))(cnn_features)
    output_l1 = Dropout(0.25)(output_l1)
    output_l1 = Dense(62, activation='softmax',kernel_initializer='he_uniform',kernel_regularizer=regularizers.l2(0.01))(output_l1)
    # classifier 2
    output_l2 = Dense(128, activation='relu',kernel_initializer='he_uniform',kernel_regularizer=regularizers.l2(0.01))(cnn_features)
    output_l2 = Dropout(0.25)(output_l2)
    output_l2 = Dense(62, activation='softmax',kernel_initializer='he_uniform',kernel_regularizer=regularizers.l2(0.01))(output_l2)
    # classifier 3
    output_l3 = Dense(128, activation='relu',kernel_initializer='he_uniform',kernel_regularizer=regularizers.l2(0.01))(cnn_features)
    output_l3 = Dropout(0.25)(output_l3)
    output_l3 = Dense(62, activation='softmax',kernel_initializer='he_uniform',kernel_regularizer=regularizers.l2(0.01))(output_l3)
    # classifier 4
    output_l4 = Dense(128, activation='relu',kernel_initializer='he_uniform',kernel_regularizer=regularizers.l2(0.01))(cnn_features)
    output_l4 = Dropout(0.25)(output_l4)
    output_l4 = Dense(62, activation='softmax',kernel_initializer='he_uniform',kernel_regularizer=regularizers.l2(0.01))(output_l4)


    model = Model(inputs=pic_in, outputs=[output_l1, output_l2, output_l3, output_l4])

    model.compile(
              optimizer=Adadelta(),
              loss='categorical_crossentropy',
              loss_weights=[1., 1.2, 1.2, 1.], 
              metrics=['accuracy'])

    return model


def train(model,train_images, train_labels,config):
    callbacks = [
        EarlyStopping(patience=5, verbose=1, mode='auto')
    ]
    if config['显示图形']:
        callbacks_history=LossHistory()
        callbacks.append(callbacks_history)
    
    if config['扩充']:
        
        def expansion_train_images(images,labels):
            i=0
            c=50
            while True:
                matrix=np.random.rand(3,3)/3
                for r in range(0,100):
                    image=images[i:i+c]
                    image=np.matmul(image,matrix)
                    yield (image,[labels[0][i:i+c],labels[1][i:i+c],labels[2][i:i+c],labels[3][i:i+c]])
                    i+=c
                    if i>=4500:
                        i=0
        def expansion_test_images(images,labels):
            i=4500
            c=50
            while True:
                for r in range(0,100):
                    image=images[i:i+c]
                    yield (image,[labels[0][i:i+c],labels[1][i:i+c],labels[2][i:i+c],labels[3][i:i+c]])
                    i+=c
                    if i>=5000:
                        i=4500
        model.fit_generator(
              expansion_train_images(train_images,train_labels),
              steps_per_epoch=800,
              epochs=config['次数'],
              batch_size=config['批量数目'],
              validation_data=expansion_test_images(train_images,train_labels),
              validation_steps=5,
              verbose=config['显示模式'],
              callbacks=callbacks,
        )
        #print(compare(model.predict(np.matmul(train_images[-500:],np.random.rand(3,3)/3)),[p[-500:] for p in train_labels]))
        print(compare(model.predict(train_images[:-500]),[p[:-500] for p in train_labels]))
        print(compare(model.predict(train_images[-500:]),[p[-500:] for p in train_labels]))
        '''
        model.fit(np.matmul(train_images,np.random.rand(3,3)/3),
              train_labels,
              epochs=config['次数'],
              batch_size=config['批量数目'],
              #validation_split=test_rate,
              verbose=config['显示模式'],
              callbacks=callbacks,
              validation_split=0.1
              )
        #print(compare(model.predict(np.matmul(train_images[:-500],np.random.rand(3,3)/3)),[p[:-500] for p in train_labels]))
        
        print(compare(model.predict(train_images[:-500]),[p[:-500] for p in train_labels]))
        print(compare(model.predict(train_images[-500:]),[p[-500:] for p in train_labels]))
        '''
    else:
        model.fit(train_images,train_labels,
              epochs=config['次数'],
              batch_size=config['批量数目'],
              #validation_split=test_rate,
              verbose=config['显示模式'],
              callbacks=callbacks,
              validation_split=0.1
              )
        print(compare(model.predict(train_images[:-500]),[p[:-500] for p in train_labels]))
        print(compare(model.predict(train_images[-500:]),[p[-500:] for p in train_labels]))
    if config['显示图形']:
        callbacks_history.loss_plot('epoch')

def predict(model,images):
    predict_tensor=model.predict(images)
    label_id_tensor=np.array(predict_tensor).argmax(axis=2)
    label=[]
    for j in range(0,label_id_tensor.shape[1]):
        label.append('')
        for i in range(0,4):
            label_number=label_id_tensor[i,j]
            label[-1]+=(int_to_base64_char(label_number))
    return label

def save_predict_result(file_names,labels,config):
    frame = pd.DataFrame({'ID':file_names,'label':labels},columns=['ID','label'])
    frame.to_csv(config['预测结果文件路径'])

def load_model(config):
    return keras.models.load_model(config['文件路径'])


def store_model(model,config):
    model.save(config['文件路径'])

if __name__ == "__main__":
    main()
