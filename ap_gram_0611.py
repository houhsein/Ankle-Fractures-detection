#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 15:28:16 2020

@author: jarjar
"""

0#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 08:36:34 2019

@author: jarjar
"""
#import tensorflow as tf 
import os
from keras.applications.densenet import DenseNet121
import cv2
from tensorflow import keras
from keras import backend as K
from keras.preprocessing import image
from keras.applications import imagenet_utils
from keras.applications.resnet50 import ResNet50, preprocess_input 
#from keras.applications.Densenet121 import decode_predictions
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
import keras.models
import json
import pandas as pd
import os
import os.path
import re
from sklearn import metrics
from sklearn.metrics import roc_auc_score


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def processing_image(img_path):
    # 讀取影像為 PIL 影像
    img = image.load_img(img_path, target_size=(1024, 1024))
    # 轉換 PIL 影像為 nparray
    x = image.img_to_array(img)
    x = x / 255
    
    # 加上一個 batch size，例如轉換 (224, 224, 3) 為 （1, 224, 224, 3) 
    x = np.expand_dims(x, axis=0)
    
    # 將 RBG 轉換為 BGR，並解減去各通道平均
#    x = preprocess_input(x)
    
    return x

def height(x, y):
	return (1-x/2+x**5+y**3)*np.exp(-x**2-y**2)


class Formatter(object): 
    def __init__(self, im): 
        self.im = im 
    def __call__(self, x, y): 
        z = self.im.get_array()[int(y), int(x)] 
        z = np.sin(np.sqrt(x ** 2 + y ** 2))
        return 'x=%i, y=%i, z=%1.4f' % (x, y, z) 


def gradcam(model, x):
    global preds, pred_class2, pred_class
    # 取得影像的分類類別
    threshold=0.5
    preds = model.predict(x)
#    pred_class2 = np.argmax(preds[0])
    if preds[0][1]>=threshold :
        
        pred_class2 = 1
    else :
        
        pred_class2 = 0
    print(preds)
    print(pred_class2)
    pred_class = 1

#    pred_class2[preds >= threshold] = 1
#    pred_class2[preds < threshold] = 0

#    print(pred_class)
    
    # 取得影像分類名稱
#    pred_class_name = imagenet_utils.decode_predictions(preds)[0][0][1]
    
    # 預測分類的輸出向量
    pred_output = model.output[:, pred_class]
    
    # 最後一層 convolution layer 輸出的 feature map
    # ResNet 的最後一層 convolution layer
    last_conv_layer = model.get_layer('conv5_block16_2_conv')
#    last_conv_layer = model.get_layer('block14_sepconv2_act')
#    block14_sepconv2_act
#    block14_sepconv2_bn
    # 求得分類的神經元對於最後一層 convolution layer 的梯度
    grads = K.gradients(pred_output, last_conv_layer.output)[0]
    
    # 求得針對每個 feature map 的梯度加總
    pooled_grads = K.sum(grads, axis=(0, 1, 2))
    
    # K.function() 讓我們可以藉由輸入影像至 `model.input` 得到 `pooled_grads` 與
    # `last_conv_layer[0]` 的輸出值，像似在 Tensorflow 中定義計算圖後使用 feed_dict
    # 的方式。
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    
    # 傳入影像矩陣 x，並得到分類對 feature map 的梯度與最後一層 convolution layer 的 
    # feature map
    pooled_grads_value, conv_layer_output_value = iterate([x])
    
    # 將 feature map 乘以權重，等於該 feature map 中的某些區域對於該分類的重要性
    for i in range(pooled_grads_value.shape[0]):
        conv_layer_output_value[:, :, i] *= (pooled_grads_value[i])
        
    # 計算 feature map 的 channel-wise 加總
    heatmap = np.sum(conv_layer_output_value, axis=-1)
    print(np.max(heatmap))
#    coorda2=np.squeeze(heatmap)
#    print(coorda2)
#    print(heatmap)
#    if preds[0]<= threshold :
#        a='normal'     
#    else :
#        a='fracture' 
#    return heatmap, a
    if pred_class2 == 0:
        a='normal'     
    else :
        a='fracture' 
    return heatmap, a


def plot_heatmap_boundingbox(heatmap, img_path, a):
    global x,y,z,s
    # ReLU
    heatmap = np.maximum(heatmap, 0)
#    aaaa=heatmap
#    coorda2=np.squeeze(heatmap)
#    print(heatmap)
#    X, Y = np.meshgrid(heatmap[0],heatmap[1])
#    print(np.max(aaaa))
    # 正規化
    heatmap /= np.max(heatmap)
#    print(np.max(heatmap))
    coorda=np.where(heatmap==np.max(heatmap))
    coorda=np.squeeze(coorda)
    print('numpy自带求最大值坐标： ',coorda)

#    print(heatmap)
    # 讀取影像
    img = cv2.imread(img_path)
    
    cv2.rectangle(img, (x,y), (s,z), (0,255,0), 5)  
    plt.figure()
    ax = plt.subplot(121)
#    cv2.rectangle(img, (int(544.2636193152624),int(1029.5435724328142)), (int(487.88911392381567),int(1219.4366432250558)), (0,255,0), 5)    
    
#    plt.figure()
#    ax = plt.subplot(121)#    fig, ax = plt.subplots(1,2,1)
    
    im = cv2.resize(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB), (img.shape[1], img.shape[0]))

    # 拉伸 heatmap
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    coorda=np.where(heatmap==np.max(heatmap))
    coorda=np.squeeze(coorda)

#    coorda=np.where(heatmap==np.max(heatmap))
#    coorda=np.squeeze(coorda)
#    (y,x)=coorda
    # 以 0.6 透明度繪製原始影像
    
    
    
    ax.imshow(im, alpha=0.6)
    


    ax=ax.imshow(heatmap, cmap='jet', alpha=0.4)
##    cv2.imwrite('fx_cv_pred_{}_{}'.format(a,i),heatmap,[int(cv2.IMWRITE_JPEG_QUALITY),95])
##    cv2.destroyAllWindows()
    ax = plt.subplot(122)
    ax.imshow(im)
    plt.imshow(img)
    plt.axis('off')
#
    plt.title('lable_'+rrr+'__pred_{}'.format(a))
##
###    plt.show()
    plt.savefig('lable_'+rrr+'__pred_{}_{}'.format(a,i),dpi=256)
    print('pltsave')
    plt.close('all')
    
    
    
##    fig, ax1 = plt.subplots() 
#    crop_img = ax1.imshow(crop_img)  
#    ax1.format_coord = Formatter(im) 

#    plt.show() 

    #fig, ax1 = plt.subplots(121)
'''    try:
        (y,x)=coorda
    #    coorda2=cv2.resize(coorda, (img.shape[1], img.shape[0]))
    #    print(x,y)     
        h=512
        w=512
        print('numpy自带求最大值坐标2： ',coorda)
        heatmap = np.uint8(255 * heatmap)
        crop_img = im[y-h:y+h, x-w:x+w] 
#        ax1.imshow(crop_img, aspect='equal')
#        cv2.imwrite('fx_pred_{}_{}'.format(a,i),crop_img,[int(cv2.IMWRITE_JPEG_QUALITY),95])
#        cv2.imwrite('crop_{}_{}'.format(a,i),crop_img,[int(cv2.IMWRITE_JPEG_QUALITY),95])
        if crop_img.size > 0:
#            cv2.imwrite("output.png", dst)#    cv2.waitKey(0)
            cv2.imwrite('crop_{}_{}'.format(a,i),crop_img,[int(cv2.IMWRITE_JPEG_QUALITY),95])

#        cap.release()
#        out.release()
            cv2.destroyAllWindows()
            plt.close()
#        plt.axis('off')
# 去除圖像周圍的白邊
#        height, width, channels = im.shape
#        # 如果dpi=300，那麼圖像大小=height*width
##        fig.set_size_inches(width/100.0/3.0, height/100.0/3.0)
#        plt.gca().xaxis.set_major_locator(plt.NullLocator())
#        plt.gca().yaxis.set_major_locator(plt.NullLocator())
#        plt.subplots_adjust(top=1,bottom=1,left=1,right=0,hspace=0,wspace=0)
#        plt.margins(0,0)
    except IOError:
        print('error')
        pass'''
        
##
###    plt.figure(figsize=(2,2))
#    plt.axis('off')
#    plt.savefig('fx__pred_{}_{}'.format(a,i),dpi=300,pad_inches=0.0)
#    plt.close('all')
    
#    cv2.imshow('image',crop_img)


#df=pd.read_csv('/home/atongb/jarjar/annotationlist0328.csv') 

 
#model = ResNet50(weights='imagenet')
#model = DenseNet121(weights='imagenet')
#model=load_model('/Users/jarjar/0227_ap_doubletrainfx_0227_1_model_seed3_categorical_60_epoch.h5')
#model=load_model('/data/home/jarjar/ankle_record/model_seed3_0109_lateraldensenet_0113_apdata_test36_512_categorical_60_epoch.h5')
#model=load_model('/data/home/jarjar/ankle_record/lateral_0223_1_model_seed3_0223_lateral_categorical_60_epoch.h5')
model=load_model('/data/jarjar/AP_and_lateral_record_20220117/fine_tune_30_epoch_categorical_crossentropy_bestdensenet121_AP_and_lateral_Ankle_mix_fold5_rotation_range02_shear_range0_2_zoom_range_0_2_imagenet_laayer5.h5')
#model2=load_model('/data/home/jarjar/ankle_record/model_seed3_0109_lateraldensenet_0113_apdata_test36_512_categorical_60_epoch.h5')
#"/Users/jarjar/PXR/Fracture/"


dirlist='fold5'
date='finallayer_AP_and_lateral_0310_lateral_rotation_range02'
modelname='densenet'

#path = "/data/home/jarjar/test_ankle_new/test/fx/"
path = "/data/jarjar/Ankle_lateral_task/"+dirlist+"/test/normal/"
#path = "/Users/jarjar/fracture/"
##path = "/data/home/jarjar/ankle_positive/test/fx/"
files= os.listdir(path)
aaa = "/data/jarjar/Ankle_lateral_task/"+dirlist+"/test/normal/"
os.makedirs("/data/jarjar//"+modelname+'_'+date+'_'+dirlist)
os.chdir("/data/jarjar/"+modelname+'_'+date+'_'+dirlist)

#fp = open("preds.txt", "a")
predww=[]
classww=[]
nameww=[]

rrr='normal'
x=int(0)
y=int(0)
z=int(0)
s=int(0)
for i in files:
    img_path = aaa+i
    img = processing_image(img_path)
    heatmap, a = gradcam(model, img)
    plot_heatmap_boundingbox(heatmap, img_path, a)
    pp1=preds[0][1]
    predww.append(preds[0][1])
    classww.append(0)
    nameww.append(i)
#
print(classww)
print(predww)
fpr, tpr, thresholds = metrics.roc_curve(classww, predww, pos_label=1)
thresholds=0.5
path = "/data/jarjar/Ankle_lateral_task/"+dirlist+"/test/fx/"
#path = "/Users/jarjar/fx/"
##path = "/data/home/jarjar/ankle_positive/test/fx/"
files= os.listdir(path)
#
#aaa = '/Users/jarjar/fx/'
##aaa = '/data/home/jarjar/try_crop/22222/'
#aaa = '/data/home/jarjar/test_ankle_new/test/fx/'
aaa = "/data/jarjar/Ankle_lateral_task/"+dirlist+"/test/fx/"

rrr='fx'
rr=0
x=int(0)
y=int(0)
z=int(0)
s=int(0)

for i in files:
    rrr = 'fx'

    img_path = aaa+i
    img = processing_image(img_path)
    heatmap, a = gradcam(model, img)
    plot_heatmap_boundingbox(heatmap, img_path, a)
    pp1=preds[0][1]
    predww.append(preds[0][1])
    classww.append(1)
    nameww.append(i)
    rr+=1
print(classww)
print(predww)

os.chdir("/data/jarjar/ROC_AP_lateral")

fp = open("preds_{}_{}_{}.csv".format(dirlist,modelname,date), "a")
for i in range(len(predww)):
    fp.write('{},{},{}\n'.format(nameww[i],predww[i],classww[i]))
fp.close()
a=roc_auc_score(classww, predww)
#
fpr, tpr, thresholds = metrics.roc_curve(classww, predww, pos_label=None)
thresholds=0.5
#
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % a)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('ROC_'+modelname+'_'+date+'_'+dirlist)


