import tensorflow as tf
import numpy as np
from model import decoder,vae
import cv2

vae.load_weights("vae_cnn.h5")
lv = np.load("lv.npy")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter("output.avi", fourcc, 30.0, (208, 120))

for i in range(1000):
    data = lv[i].reshape(1,128)
    img = decoder.predict(data)
    img = np.array(img).reshape(120,208,1)
    img = img * 255
    img = np.array(img).astype("uint8")
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    video.write(img)
video.release()
