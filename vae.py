#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 18:10:49 2020

@author: marla
"""


'''
Write a VAE to generate mnist images
'''

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Reshape, Conv2DTranspose, BatchNormalization, LeakyReLU, Dropout, Input, Lambda, Activation
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import random
import numpy as np
import matplotlib.pyplot as plt


#load and scale data
def getData():
    (x_train, _), (x_test, _)=mnist.load_data()
    #add channel dimension
    x_train = np.expand_dims(x_train,-1)
    #scale to range -1 to 1
    x_train = (x_train /255.0)
    x_train = x_train.astype('float32')
    
    return x_train

def displayImages(images, imgTitle):
    # scale from [-1,1] to [0,1]
	images = (images + 1) / 2.0
	# plot images
	for i in range(49):
		# define subplot
		plt.subplot(7,7, 1 + i)
		# turn off axis
		plt.axis('off')
		# plot raw pixel data
		plt.imshow(np.around(np.squeeze(images[i]),2), cmap='gray')
	plt.suptitle(imgTitle)
    #show
	plt.show()
	plt.close()
    
def define_vae(input_dim, latent_dim, x_train):
    #use functional api instead of sequential model; specify the new layer with the input
    #as the previous layer in () after
    encoder_input = Input(shape=input_dim)
    x = (Conv2D(32, kernel_size = 3, strides = 1, padding = 'same'))(encoder_input)
    x = LeakyReLU()(x)
    
    x = (Conv2D(64, kernel_size = 3, strides = 2, padding = 'same'))(x)
    x = LeakyReLU()(x)
    
    x = (Conv2D(64, kernel_size = 3, strides = 2, padding = 'same'))(x)
    x = LeakyReLU()(x)
    
    x = (Conv2D(64, kernel_size = 3, strides = 1, padding = 'same'))(x)
    x = LeakyReLU()(x)
    
    #flatten and output to latent space dim
    x=Flatten()(x)
    
    #flatten goes to the mu and log_var layers that define a continuous latent space
    mu = Dense(latent_dim)(x)
    log_var = Dense(latent_dim)(x)
    
    #map to latent space by randomly sample from the normal distribution mean 0; std dev 1
    encoder_output = Lambda(sample)([mu, log_var])

    #create model from input and output layers
    encoder = Model(encoder_input, encoder_output)
   
    
    #DECODER
    decoder_input = Input(shape = latent_dim)
    #the encoder architecture ens with a flattened layer of 3136 nodes for a 7x7x64 image
    x=Dense(3136)(decoder_input)
    x=Reshape((7,7,64))(x)
    
    x=(Conv2DTranspose(filters = 64,kernel_size = 3,strides = 1,padding='same'))(x)
    x=(LeakyReLU())(x)
    
    x=(Conv2DTranspose(filters = 64,kernel_size = 3,strides = 2,padding='same'))(x)
    x=(LeakyReLU())(x)
    
    x=(Conv2DTranspose(filters = 32,kernel_size = 3,strides = 2,padding='same'))(x)
    x=(LeakyReLU())(x)
    
    x=(Conv2DTranspose(filters = 1,kernel_size = 3,strides = 1,padding='same', activation = 'sigmoid'))(x)
    
    decoder_output = x
    decoder = Model(decoder_input, decoder_output)
    
    #VAE
    vae_input = encoder_input
    vae_output = decoder(encoder_output)
    vae = Model(vae_input, vae_output)
    
    
    #TRAIN
     #displayImages(X, 'Raw Data')
    input_dim = x_train[0].shape
   # encoder = define_autoencoder(input_dim, latent_dim)
    #decoder = define_decoder(latent_dim)
    
   
    optimizer = tf.keras.optimizers.Adam()
    vae.compile(optimizer = optimizer, loss = 'binary_crossentropy')
    
    vae.fit(x=x_train, y=x_train, batch_size = 128, shuffle=True, epochs = 2)
    
    
    # Display a 2D manifold of the digits
    n = 15  # figure with 15x15 digits
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # We will sample n points within [-15, 15] standard deviations
    grid_x = np.linspace(-15, 15, n)
    grid_y = np.linspace(-15, 15, n)

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            print("Z ", z_sample)
            x_decoded = decoder.predict(z_sample)
            print("x shape ", x_decoded.shape)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            print("digit shape ", digit.shape)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure)
    plt.show()
    
    
    
def sample(args):
    mu, log_var = args
    #print(args)
    #randomly sample from normal distribution using the equation:
    #z = mu + sigma * epsilon
    # where sigma = exp(log_var/2)
    epsilon = tf.random.normal(tf.shape(mu), 0.0, 1.0, tf.float32)
    sigma = tf.math.exp(log_var/2.0)
    z = mu + sigma * epsilon
    return z
    

def define_decoder(latent_dim):
    input_layer = Input(shape = latent_dim)
    #the encoder architecture ens with a flattened layer of 3136 nodes for a 7x7x64 image
    x=Dense(3136)(input_layer)
    x=Reshape((7,7,64))(x)
    
    x=(Conv2DTranspose(filters = 64,kernel_size = 3,strides = 1,padding='same'))(x)
    x=(LeakyReLU())(x)
    
    x=(Conv2DTranspose(filters = 64,kernel_size = 3,strides = 2,padding='same'))(x)
    x=(LeakyReLU())(x)
    
    x=(Conv2DTranspose(filters = 32,kernel_size = 3,strides = 2,padding='same'))(x)
    x=(LeakyReLU())(x)
    
    x=(Conv2DTranspose(filters = 1,kernel_size = 3,strides = 1,padding='same', activation = 'sigmoid'))(x)
    
    
    model = Model(input_layer, x)
    #model.summary()
    return model
    
'''
def define_vae(encoder, decoder):
    model_input = encoder.layers[0]
    encoder_output = encoder.layers[len(encoder.layers)-1]
    model_output = decoder(encoder_output)
    model = Model(model_input, model_output)
    model.summary()
    return model
'''

def rmse_loss(y_true, y_pred):
    return tf.math.reduce_mean(tf.square(y_true-y_pred), axis = [1,2,3])

'''
def train(x_train, latent_dim):
    #displayImages(X, 'Raw Data')
    input_dim = x_train[0].shape
   # encoder = define_autoencoder(input_dim, latent_dim)
    #decoder = define_decoder(latent_dim)
    
    vae = define_vae(input_dim, latent_dim)
    vae.summary()
    
    optimizer = tf.keras.optimizers.Adam(lr = 0.002)
    vae.compile(optimizer = optimizer, loss = rmse_loss)
    
    vae.fit(x=x_train, y=x_train, batch_size = 32, shuffle=True, epochs = 10)
    
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure)
    plt.show()
    
    return vae
'''    
X = getData()
latent_dim = 2
tf.random.set_seed(1)
input_dim = X[0].shape
define_vae(input_dim, latent_dim, X[:10000,:,:,:])





    
    
