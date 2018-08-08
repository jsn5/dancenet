from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Reshape
from keras.layers import Input
from keras.layers import UpSampling2D
from keras.layers import Lambda
from keras.models import Model
from keras.losses import binary_crossentropy
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import numpy as np
import cv2
import os

latent_dim = 128
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


input_img = Input(shape=(120,208,1))
x = Conv2D(filters=128,kernel_size=3, activation='relu', padding='same')(input_img)
x = MaxPooling2D(pool_size=2)(x)
x = Conv2D(filters=64,kernel_size=3, activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=2)(x)
x = Conv2D(filters=32,kernel_size=3, activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=2)(x)
shape = K.int_shape(x)
x = Flatten()(x)
x = Dense(128,kernel_initializer='glorot_uniform')(x)

z_mean = Dense(latent_dim)(x)
z_log_var = Dense(latent_dim)(x)
z = Lambda(sampling, output_shape=(latent_dim,), name="z")([z_mean,z_log_var])

encoder = Model(input_img, [z_mean, z_log_var,z], name="encoder")
encoder.summary()


latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(shape[1] * shape[2] * shape[3], kernel_initializer='glorot_uniform',activation='relu')(latent_inputs)
x = Reshape((shape[1],shape[2],shape[3]))(x)
x = Dense(128,kernel_initializer='glorot_uniform')(x)
x = Conv2D(filters=32, kernel_size=3, activation='relu', padding='same')(x)
x = UpSampling2D(size=(2,2))(x)
x = Conv2D(filters=64,kernel_size=3, activation='relu', padding='same')(x)
x = UpSampling2D(size=(2,2))(x)
x = Conv2D(filters=128,kernel_size=3, activation='relu', padding='same')(x)
x = UpSampling2D(size=(2,2))(x)
x = Conv2D(filters=1,kernel_size=3, activation='sigmoid', padding='same')(x)

decoder = Model(latent_inputs,x,name='decoder')

decoder.summary()


outputs = decoder(encoder(input_img)[2])
print(outputs.shape)
vae = Model(input_img,outputs,name="vae")

def data_generator(batch_size,limit):

	batch = []
	counter = 1
	while 1:
		for i in range(1,limit+1):
			if counter >= limit:
				counter = 1
			img = cv2.imread("imgs/{}.jpg".format(counter),cv2.IMREAD_GRAYSCALE)
			img = img.reshape(120,208,1)
			batch.append(img)
			if len(batch) == batch_size:
				batch_np = np.array(batch) / 255
				batch = []
				yield (batch_np,None)
			counter += 1

if __name__ == '__main__':

	reconstruction_loss = binary_crossentropy(K.flatten(input_img), K.flatten(outputs))
	reconstruction_loss *= 120 * 208
	kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
	kl_loss = K.sum(kl_loss,axis=-1)
	kl_loss *= -0.5
	vae_loss = K.mean(reconstruction_loss + kl_loss)
	vae.add_loss(vae_loss)
	vae.compile(optimizer='rmsprop', loss = None)
	vae.summary()
	checkpoint = ModelCheckpoint('./weights/vae_cnn.h5', verbose=1,monitor='loss', save_best_only=True, mode='auto',period=1)
	callbacks= [checkpoint]
	batch_size = 100
	limit = len(os.listdir('imgs'))
	spe = int(limit/batch_size)
	print(limit,spe)
	vae.fit_generator(data_generator(batch_size,limit),epochs=50, steps_per_epoch=spe,callbacks=callbacks)
	vae.save_weights('vae_cnn.h5')
