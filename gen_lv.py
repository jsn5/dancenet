import tensorflow as tf
from tensorflow.contrib.keras import backend as K
import os
import numpy as np
import cv2
from model import encoder,decoder,vae

vae.load_weights('vae_cnn.h5')
data =[]
lv_array = []
limit = len(os.listdir('imgs'))

for i in range(1,limit):
	img = cv2.imread('imgs/{}.jpg'.format(i),cv2.IMREAD_GRAYSCALE)
	img = cv2.resize(img,(208,120))
	data_np = np.array(img) / 255
	data_np = data_np.reshape(1,120,208,1)
	lv = encoder.predict(data_np)[2]
	lv_array.append(lv)

print(np.array(lv_array).shape)
np.save("lv.npy",np.array(lv_array))





