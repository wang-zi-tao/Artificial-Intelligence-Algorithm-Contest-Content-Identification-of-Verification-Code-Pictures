import glob
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from keras.preprocessing import image


train_dir="./验证码图片内容识别竞赛数据/train/"
test_dir="./验证码图片内容识别竞赛数据/train/"
#加载图片和验证码文字
def load_picture_and_label(images_path=train_dir):
    images=[]
    labels=[]
    pd_data=pd.read_csv(images_path+'train_label.csv')
    for i in range(0,pd_data.shape[0]):
        image_path=pd.iloc[i,0]
        label=pd.iloc[i,1]
        image_data = image.load_img(image_path)
        images.append(image.img_to_array(images))
        labels.append(label)
    return images,labels
