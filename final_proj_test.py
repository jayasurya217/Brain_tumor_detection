# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 14:17:45 2021

@author: jayas
"""


from tensorflow.keras.models import load_model
from keras.preprocessing import image
import numpy as np
import tensorflow as tf 

model=load_model(r"C:\KEC\AI & ML\Final_Project\brain_tumor_dataset\brain.h5")
img=image.load_img(r"C:\KEC\AI & ML\Final_Project\brain.jpeg", target_size=(128,128))
x=image.img_to_array(img)
print(x)
print(x.shape)

x=np.expand_dims(x, axis=0)
print(x.shape)

#pred=model.predict_classes(x)
y=model.predict(x)
pred= np.argmax(y, axis=1)
print(pred)