#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 09:16:11 2019

@author: jarjar
"""

import tensorflow as tf 
import os
import tensorflow
import csv
import pandas as pd 
import matplotlib.pyplot as plt
#import utils
#import matplotlib.image as mpimg
from PIL import Image
from keras.preprocessing import image
import cv2
import time
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import keras
import numpy as np
import datetime
from keras.models import Sequential
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Dense, Input, Dropout
from keras.applications.densenet import DenseNet121
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.layers import Activation
from keras.layers import Conv2D, GlobalAveragePooling2D, Convolution1D, MaxPooling2D
from keras.layers.core import Dense, Flatten, SpatialDropout1D, SpatialDropout2D
from keras.optimizers import Adam, RMSprop
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
import keras.callbacks
from keras.models import load_model
import itertools
from PIL import Image
from keras.callbacks import ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from collections import Counter
import sys
from keras import initializers
from keras import regularizers
from sklearn.metrics import classification_report, confusion_matrix
from keras.applications import Xception

def img_show2(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

def show_train_acc_history(train_acc,validation):
    plt.plot(train_history.history[train_acc])
    plt.plot(train_history.history[validation])
    plt.title('Train_acc History')
    plt.ylabel('Train')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    
    
def show_train_loss_history(train_acc,validation):
    plt.plot(train_history.history[train_acc])
    plt.plot(train_history.history[validation])
    plt.title('Train_loss History')
    plt.ylabel('Train')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
def focal_loss(gamma=2., alpha=.25):

	def focal_loss_fixed(y_true, y_pred):
		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
		return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1+K.epsilon())) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
	return focal_loss_fixed

  
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

img_train=[]
img_test=[]
count=0
count2=0
img_sum=[]
img_name=[]
test_label=[]
train_label=[]

train_dir = '/data/jarjar/Ankle_mix/fold3/train/'
test_dir = '/data/jarjar/Ankle_mix/fold3/test/'
val_dir = '/data/jarjar/Ankle_mix/fold3/val/'

#/data/home/jarjar/ankle_positive
trainfiles = os.listdir(train_dir)
testfiles = os.listdir(test_dir)

img_size=1024
num_epochs = 1
batch_size = 4
test_datagen = ImageDataGenerator(rescale=1. / 255)
#rescale=1. / 255
validation_datagen = ImageDataGenerator(
        rescale=1./255,
		rotation_range=20,
		zoom_range=0.2,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.2,
		horizontal_flip=True)


train_datagen = ImageDataGenerator(
        rescale=1./255,
		rotation_range=20,
		zoom_range=0.2,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.2,
		horizontal_flip=True)
       
train_generator = train_datagen.flow_from_directory(train_dir,
                                               target_size=(img_size, img_size),
                                               color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='categorical',
                                               classes=['normal', 'fx'])
test_generator = test_datagen.flow_from_directory(test_dir,
                                               target_size=(img_size, img_size),
                                               color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='categorical',
                                               classes=['normal', 'fx'])
validation_generator = validation_datagen.flow_from_directory(val_dir,
                                               target_size=(img_size, img_size),
                                               color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='categorical',
                                               classes=['normal', 'fx'])

#training_generator, steps_per_epoch = balanced_batch_generator(X,y,sampler=NearMiss(), 
#                                                               batch_size=10, random_state=42)
testfilenames = test_generator.filenames
trainfilenames = train_generator.filenames
validationfilenames = validation_generator.filenames
nb_train_samples = len(trainfilenames)
nb_test_samples = len(testfilenames)
nb_validation_samples = len(validationfilenames)
#steps_per_epoch=int((len(train_generator.filenames)*3)/batch_size)
#print(steps_per_epoch)
counter = Counter(train_generator.classes)                          
max_val = float(max(counter.values()))       
#class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()} 
class_weights = {0: 1, 1: 1}
print(class_weights)
base_model = DenseNet121(input_shape=(img_size, img_size, 3),
                         weights='imagenet',
                         include_top=False, pooling='avg')
base_model.layers.pop()
#new_model = tf.keras.models.Sequential(base_model.layers[:-1])
model = base_model.output
relu_act = Dense(1024 ,activation='relu')(model)
dro = Dropout(0.5 ,name='drop')(relu_act)
predictions = Dense(2, activation='softmax', name = 'predictions' )(dro)
model_den = Model(inputs=base_model.input, outputs=predictions, name='den121')
model_den.summary()

LR_function=ReduceLROnPlateau(monitor='val_loss',
                             patience=5,       # 3 epochs 內loss沒下降就要調整LR
                             verbose=1,
                             factor=0.05,  # LR降為0.5
                             min_lr=0.000001)

filepath="60epoch_categorical_crossentropy_bestdensenet121_AP_and_lateral_Ankle_mix_fold3_rotation_range02_shear_range0_2_zoom_range_0_2_imagenet.h5"

#os.chdir("/data/home/jarjar/0113_record/")
os.chdir("/data/jarjar/AP_and_lateral_record_20220117")
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True,
mode='min')
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"Ankle_mix_fold3_rotation_range02_shear_range0_2_zoom_range_0_2_categorical_crossentropy_densenet_imagenet")
callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, write_graph=True, write_images=True)


#adam=Adam(lr=0.00005)
#relu
#softmax
#sigmoid

INIT_LR = 0.00001

adam=Adam(lr=INIT_LR)
#model_den.compile(optimizer=adam, loss=[focal_loss(alpha=2.0, gamma=0.3)], metrics=['accuracy'])
model_den.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

steps_per_epoch=int((nb_train_samples)/batch_size)

epochs=60
train_history=model_den.fit_generator(train_generator, callbacks=[LR_function,callback,checkpoint]
, epochs=epochs,verbose=1, validation_data=validation_generator, validation_steps=nb_validation_samples//(batch_size), 
steps_per_epoch=steps_per_epoch)
#class_weight=class_weights


losses = train_history.history['loss']
val_losses = train_history.history['val_loss']
acc = train_history.history['acc']
val_acc = train_history.history['val_acc']
print(losses,val_losses)

test_loss, test_acc = model_den.evaluate_generator(test_generator, steps=nb_test_samples)
print('test acc:', test_acc)
#prediction=model_den.predict_generator(test_generator,verbose=1,steps=nb_test_samples)
#print(prediction)
#prediction=prediction[1]
#predict_label = list(lambda x: 0 if x<0.5 else 1, prediction)
#predict_label=np.argmax(prediction,axis=1)
#print(predict_label)
#true_label=test_generator.classes
#print(true_label)
#print(pd.crosstab(true_label,predict_label,rownames=['label'],colnames=['predict']))
#Y_pred = model_den.predict_generator(test_generator, nb_test_samples // batch_size+1)
#y_pred = np.argmax(Y_pred, axis=1)
#print('Confusion Matrix')
#print(confusion_matrix(test_generator.classes, y_pred))
#print('Classification Report')
#target_names = ['ap_normal', 'ap_fracture']
#print(classification_report(test_generator.classes, y_pred, target_names=target_names))
model_den.save('Ankle_mix_fold3_rotation_range02_change_zoom_range_0_2_shear_range0_2_imagenet_model_categorical_crossentropy_60_epoch_relu_softmax.h5')

print ('run finish')

