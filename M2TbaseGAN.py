# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 09:54:50 2019

@author: Rohit Gandikota
"""

import keras 
from keras.models import Model
from keras.layers import *
from keras.activations import *
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import plot_model
import matplotlib.pyplot as plt
import os
import numpy as np
from data_loader import load_data, load_batch
import datetime
import tensorflow as tf
from keras.backend import tensorflow_backend as K
from keras.optimizers import Adam
import random

def sample_images(epoch,batch,gen,enc):
    r, c = 3, 3

    imgs_A, imgs_B = load_data(batch_size=3, is_testing=True)
    imgs_B_lat = enc.predict(imgs_B)
    fake_A = gen.predict(imgs_B_lat)
 

    # Rescale images 0 - 1
    titles = ['Condition', 'Generated', 'Original']
    fig, axs = plt.subplots(r, c)
    
    for i in range(r):
        for j in range(c):
            if j == 0:
                axs[i,j].imshow(imgs_B[i])
            if j== 1:    axs[i,j].imshow(np.reshape(fake_A[i], (256,256)), cmap='gray')
            if j == 2:     axs[i,j].imshow(np.reshape(imgs_A[i], (256,256)), cmap='gray')
            axs[i, j].set_title(titles[j])
            axs[i,j].axis('off')
        
    fig.savefig(f"Auto\\Test_Results_Roads\\{epoch}-{batch}.png")
    plt.close()


def ResEncoder(inp, nfilter, ksize=3, stride=2, bn = True):
    
    x = Conv2D(int(nfilter/2), ksize, strides = 1, padding= 'same')(inp)
    x = LeakyReLU()(x)    
    x = BatchNormalization()(x)
    
    x = Conv2D(nfilter, ksize, strides = stride, padding= 'same')(x)   
    x = LeakyReLU()(x)  
    x = BatchNormalization()(x)
    
    x1 = Conv2D(nfilter, 3, strides = stride, padding='same')(inp)
    x1 = LeakyReLU()(x1)  
    x1 = BatchNormalization()(x1)
    
    out = Add()([x,x1])
    
    return out


def DiscEncoder(inp, nfilter, ksize=3, stride=2, bn = True):
    
    x = Conv2D(int(nfilter/2), ksize, strides = 1, padding= 'same')(inp)
    x = LeakyReLU()(x)    
    x = BatchNormalization()(x)
    
    x = Conv2D(nfilter, ksize, strides = stride, padding= 'same')(x)   
    x = LeakyReLU()(x)  
    x = BatchNormalization()(x)
    
    x1 = Conv2D(nfilter, 3, strides = stride, padding='same')(inp)
    x1 = LeakyReLU()(x1)  
    x1 = BatchNormalization()(x1)
    
    
    return x1

def ResDecoder(inp, enc_layer, nfilter, ksize=3, stride=2, bn = True):
    
    x = UpSampling2D(size=(2, 2))(inp)
    x = Conv2DTranspose(int(nfilter/2), ksize, strides=(1, 1), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    
    x = Conv2DTranspose(nfilter, ksize, strides=(1, 1), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    
    x1 = UpSampling2D(size=(2, 2))(inp)
    x1 = Conv2DTranspose(nfilter, ksize, strides=(1, 1), padding='same', activation='relu')(x1)
    x1 = BatchNormalization()(x1)
    
    out = Add()([x,x1])
    skip_out = Concatenate()([out,enc_layer])
    
    return skip_out

def Decoder(inp,  nfilter, ksize=3, stride=2, bn = True):
    
    x = UpSampling2D(size=(2, 2))(inp)
    x = Conv2DTranspose(int(nfilter/2), ksize, strides=(1, 1), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    
    x = Conv2DTranspose(nfilter, ksize, strides=(1, 1), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    
    x1 = UpSampling2D(size=(2, 2))(inp)
    x1 = Conv2DTranspose(nfilter, ksize, strides=(1, 1), padding='same', activation='relu')(x1)
    x1 = BatchNormalization()(x1)
    
    out = Add()([x,x1])
    
    return out

#%% Building Generator
def build_gen():
    nf = 8
    inp_layer1 = Input(shape=(4,4,256))

    # Decoder
    x6 = Decoder(inp_layer1,  nf*16)
    x7 = Decoder(x6, nf*8)
    x8 = Decoder(x7, nf*4)
    x9 = Decoder(x8, nf*2)
    x10= Decoder(x9, nf)
    
    x11 = UpSampling2D(size=(2, 2))(x10)
    out = Conv2DTranspose(1, 3 , strides=(1, 1), padding='same', activation='sigmoid')(x11)
    gen = Model(inp_layer1 , out)
    
#    plot_model(gen, to_file="C:\\Users\\user\\Desktop\\Projects\\ImageTileFD\\Test_Results\\Generator.png",show_shapes=True, show_layer_names=True)
    return gen
#%% Building Discriminator
def build_disc():
    nf = 4
    img_A = Input(shape=(256,256,1))
    img_B = Input(shape=(256,256,1))
    # Concatenate image and conditioning image by channels to produce input
    combined_imgs = Concatenate(axis=-1)([img_A, img_B])
    x = DiscEncoder(combined_imgs, nf) # 64
    x = DiscEncoder(x, nf*2) # 32 
    x = DiscEncoder(x, nf*4) # 16
    x = DiscEncoder(x, nf*8) # 8
    x = DiscEncoder(x, nf*16) # 4
    x = DiscEncoder(x, nf*32) # 2 
    x = DiscEncoder(x, nf*64) # 2 
    x = DiscEncoder(x, nf*64) # 2 
    x = Flatten()(x)
    x = Dense(1, activation = 'sigmoid')(x)
    
    disc = Model([img_A,img_B], x)
    
    return disc


def build_auto():
    nf = 8
    inp_layer1 = Input(shape=(256,256,3))
    # Encoder 
    x  = ResEncoder(inp_layer1, nf) # 64
    x1 = ResEncoder(x, nf*2) # 32 
    x2 = ResEncoder(x1, nf*4) # 16
    x3 = ResEncoder(x2, nf*8) # 8
    x4 = ResEncoder(x3, nf*16) # 4
    x5 = ResEncoder(x4, nf*32) # 2 
    # Decoder
    x6 = ResDecoder(x5, x4, nf*16)
    x7 = ResDecoder(x4, x3, nf*8)
    x8 = ResDecoder(x7, x2, nf*4)
    x9 = ResDecoder(x8, x1, nf*2)
    x10 = ResDecoder(x9, x, nf)
    
    x11 = UpSampling2D(size=(2, 2))(x10)
    out = Conv2DTranspose(3, 3 , strides=(1, 1), padding='same', activation='sigmoid')(x11)
    encoder = Model(inp_layer1, x5)
    auto    = Model(inp_layer1, out)
    return encoder, auto

#%% Autoencoder training
with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=50)) as sess:
    K.set_session(sess)
    epochs = 2000
    optimizer = Adam(0.002, 0.5)
    batch_size = 100
    # Build and compile the discriminator
    enc,auto = build_auto()
    auto.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
    start_time = datetime.datetime.now()
    for epoch in range(epochs):
        for batch_i, (water_data, sat_data) in enumerate(load_batch(batch_size = batch_size)):
    
            # ---------------------
            #  Train Autoencoder
            # ---------------------

            a_loss = auto.train_on_batch(sat_data, sat_data)
            elapsed_time = datetime.datetime.now() - start_time
            print ("[Epoch %d/%d] [Batch %d] [A loss: %f, acc: %3d%%] time: %s" % (epoch, epochs, batch_i,
                                                                    a_loss[0], 100*a_loss[1],
                                                                    elapsed_time))
            if batch_i%50 == 0:
                num = random.randint(0,batch_size-1)
                w=5
                h=5
                fig=plt.figure(figsize=(8, 8))
                columns = 2
                rows = 1
                for i in range(1, columns*rows +1):
                    fig.add_subplot(rows, columns, i)
                    img = sat_data[num]
                    pred = auto.predict(img.reshape(1,256,256,3))
                    if i == 1:
                        plt.imshow(np.uint(pred.reshape(256,256,3)*255))
                    else:
                        plt.imshow(np.uint(img*255))
                fig.savefig(f"Auto\\Test_Results\\{epoch}-{batch_i}.png")
                plt.close(fig)
        auto.save('C:\\Users\\user\\Desktop\\Projects\\ImageTileFD\\Auto\\Models\\auto.h5')
#%% GAN training
        
with tf.Session(config=tf.ConfigProto(
                intra_op_parallelism_threads=50)) as sess:
    K.set_session(sess)
    
    optimizer = Adam(0.0002, 0.5)
    optimizer1 = Adam(0.002, 0.5)
    # Build and compile the discriminator
    disc = build_disc()
    disc.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
    
    
    gen = build_gen()
    
    #-------------------------
    # Construct Computational
    #   Graph of Generator
    #-------------------------
    
    # Build the generator

    # Input images and their conditioning images
    water = Input(shape=(256,256,1))
    sat = Input(shape=(256,256,3))
    sat_lat = Input(shape=(4,4,256))
    
    # By conditioning on B generate a fake version of A
    fake_water = gen(sat_lat)
    
    # For the combined model we will only train the generator
    disc.trainable = False
    
    # Discriminators determines validity of translated images / condition pairs
    valid = disc([fake_water, water])
    
    combined = Model(inputs=[water, sat_lat], outputs=[valid, fake_water])
    combined.compile(loss=['mse', 'mse'], loss_weights=[1, 10], optimizer=optimizer1)
    
    epochs = 1000
    sample_interval = 50
    
    batch_size = 100
    
    # Adversarial loss ground truths
    valid = np.ones((batch_size,))
    fake = np.zeros((batch_size,))
    
    print('Training started')
    start_time = datetime.datetime.now()
    for epoch in range(epochs):
        for batch_i, (water_data, sat_data) in enumerate(load_batch(batch_size = batch_size)):
    
            # ---------------------
            #  Train Discriminator
            # ---------------------
            sat_data_lat = enc.predict(sat_data)
            # Condition on B and generate a translated version
            fake_water_data = gen.predict(sat_data_lat)
            # Train the discriminators (original images = real / generated = Fake)
            d_loss_real = disc.train_on_batch([water_data, water_data], valid)
            d_loss_fake = disc.train_on_batch([fake_water_data, water_data], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    
            # -----------------
            #  Train Generator
            # -----------------
    
            # Train the generators
            g_loss = combined.train_on_batch([water_data, sat_data_lat], [valid, water_data])
            g_loss = combined.train_on_batch([water_data, sat_data_lat], [valid, water_data])
    
            elapsed_time = datetime.datetime.now() - start_time
            # Plot the progress
            print ("[Epoch %d/%d] [Batch %d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs, batch_i,
                                                                    d_loss[0], 100*d_loss[1],
                                                                    g_loss[0],
                                                                    elapsed_time))
    
            # If at save interval => save generated image samples
            if batch_i % sample_interval == 0:
                sample_images(epoch= epoch, batch = batch_i,gen = gen, enc=enc)
