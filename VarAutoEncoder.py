#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 13:11:27 2020

@author: marla
"""
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Reshape, Conv2DTranspose, BatchNormalization, LeakyReLU, Dropout, Input, Lambda, Activation, Layer
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from tensorflow.keras.losses import Loss


class Sample(tf.keras.layers.Layer):
    def __init__(self):
        super(Sample, self).__init__()
        
    def build(self, input_shape):
        _, sigma_shape = input_shape
        self.sigma_shape = (sigma_shape[-1], )
        
    def call(self, inputs):
        mu, log_var = inputs
        
        #add loss 
        kl_loss = -0.5 * K.sum(1 + log_var - K.square(mu) - K.exp(log_var), axis = 1)
        kl_loss = K.mean(kl_loss, axis = 0)
      #  print("LOSSSSSS ", kl_loss)
        self.add_loss(kl_loss, inputs = inputs)
        
        #return sammple
        epsilon = tf.random.normal(tf.shape(mu), 0.0, 1.0, tf.float32)
        sigma = tf.math.exp(log_var/2.0)
        z = mu + sigma * epsilon
        return z
        
        

class VarAutoEncoder(object):
    def __init__(self, latent_dim, input_dim):
        self.latent_dim = latent_dim
        self.input_dim = input_dim
       
        
    def sample(self, args):
        mu, log_var = args
        #print(args)
        #randomly sample from normal distribution using the equation:
        #z = mu + sigma * epsilon
        # where sigma = exp(log_var/2)
        epsilon = tf.random.normal(tf.shape(mu), 0.0, 1.0, tf.float32)
        sigma = tf.math.exp(log_var/2.0)
        z = mu + sigma * epsilon
        return z
        
    def define_encoder(self):
         #use functional api instead of sequential model; specify the new layer with the input
        #as the previous layer in () after
        encoder_input = Input(shape=self.input_dim)
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
        mu = Dense(self.latent_dim, name='mu')(x)
        log_var = Dense(self.latent_dim, name='log_var')(x)
        
        #add the custom loss layer
        #mu, log_var = KLDivergenceLayer()([mu, log_var])
          
        #the encoder should output the encoded latent space vector z
        #encoder_output = Lambda(self.sample)([mu, log_var])
        encoder_output = Sample()([mu, log_var])
        #create model from input and output layers
        encoder = Model(encoder_input, encoder_output)
        encoder.summary()
        
        #self.encoder = encoder
       # self.encoder_input = encoder_input
       # self.encoder_output = encoder_output
        
        #print("input ", self.encoder_input)
        return encoder, encoder_input, encoder_output, log_var, mu
    
    def define_decoder(self):
        #DECODER
        decoder_input = Input(shape = self.latent_dim)
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
        
        #self.decoder = decoder
        return decoder
    
    def define_vae(self, encoder, decoder, encoder_input, encoder_output, log_var, mu):
         #VAE
         vae_input = encoder_input
         vae_output = decoder(encoder_output)
         vae = Model(vae_input, vae_output)
         optimizer = tf.keras.optimizers.Adam()
         #vae.compile(optimizer = optimizer, loss = 'binary_crossentropy')
         vae.compile(optimizer = optimizer, loss = self.rmse_loss)
         #self.vae = vae
         return vae

     
    def param_loss(self, log_var, mu):
        kl_loss = -0.5 * K.sum(1 + log_var - K.square(mu) - K.exp(log_var), axis=0)
        def loss_function(y_true, y_pred):
            print("do i have ", y_true.shape)
            
           
            print("KL ", kl_loss)
            rmse_loss = K.mean(K.square(y_true - y_pred), axis = [1,2,3])
            print("RMSE ", rmse_loss)
            print(K.mean(rmse_loss, axis=0))
            print(K.mean(kl_loss, axis=0))
            loss = K.mean(kl_loss, axis=0) + K.mean(rmse_loss, axis=0)
            print("LOSS ", loss)
            return loss
        return loss_function
    
    def rmse_loss(self, y_true, y_pred):
     #   print("rmse shape ", y_true.shape)
      #  print("rmse shape pred ", y_pred.shape)
        #will use rmse as reconstruction loss term
        sq=tf.math.square(y_true - y_pred)
        rmse_loss = tf.math.reduce_mean(sq, axis = [1,2,3])
      #  print("loss shape ", rmse_loss.shape)
        loss = K.mean(rmse_loss, axis = 0)
        #print("loss 1 ", loss.shape)
        return  1000*loss
        
    
    def kl_loss(self, y_true, y_pred, log_var, mu):
        #equation: kl_loss = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        print(-0.5 * tf.math.reduce_sum(1 + log_var - mu**2 - tf.math.exp(log_var)))
        return -0.5 * tf.math.reduce_sum(1 + log_var - mu**2 - tf.math.exp(log_var))
    
    def train_model(self, xtrain, vae):
         vae.fit(x=x_train, y=x_train, batch_size = 128, shuffle=True, epochs = 4)
         
    
    
    def generateImages(self, decoder):
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
                
                x_decoded = decoder.predict(z_sample)
                
                digit = x_decoded[0].reshape(digit_size, digit_size)
                #print("digit shape ", digit.shape)
                figure[i * digit_size: (i + 1) * digit_size,
                       j * digit_size: (j + 1) * digit_size] = digit
    
        plt.figure(figsize=(10, 10))
        plt.imshow(figure)
        plt.show()
        
    def createImages(self, decoder):
        #for 2D z-space
        dim_1 = np.array([random.uniform(-15, 15) for x in range(25)])
        dim_2 = np.array([random.uniform(-15, 15) for x in range(25)])
       
        z_space = np.column_stack((dim_1, dim_2))
        
      
        for i in range(len(z_space)):
            x_decoded = decoder.predict(np.expand_dims(z_space[i], axis=0))
            x_decoded = x_decoded.reshape(28,28)
            plt.subplot(5,5,i+1)
            plt.imshow(x_decoded)
            plt.axis('off')  
        plt.tight_layout(pad=0.05, w_pad=0.05, h_pad=0.05)
        plt.show()
        plt.close()
        
    
 

  
class MNISTImages(object):
    def __init__(self):
        self.x_train = self.loadTrainImages()
        
    def loadTrainImages(self):
        print('loading')
        (x_train, _), (x_test, _)=mnist.load_data()
        #add channel dimension
        x_train = np.expand_dims(x_train,-1)
        #scale to range -1 to 1
        x_train = (x_train /255.0)
        x_train = x_train.astype('float32')
        
        return x_train
        
    def getTrainImages(self):
        return self.x_train
    
    def displayImages(self, imgTitle):
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
  
 
tf.config.experimental_run_functions_eagerly(True)   
tf.random.set_seed(1)
latent_dim = 2

TrainImages = MNISTImages()
x_train = TrainImages.getTrainImages()
#x_train=x_train[:5000,:,:,:]
print("x ", x_train.shape)
input_dim = x_train[0].shape

VAE = VarAutoEncoder(latent_dim, input_dim)
encoder, encoder_input, encoder_output, log_var, mu = VAE.define_encoder()
decoder = VAE.define_decoder()
vae = VAE.define_vae(encoder, decoder, encoder_input, encoder_output, log_var, mu)

VAE.train_model(x_train, vae)
VAE.createImages(decoder)


