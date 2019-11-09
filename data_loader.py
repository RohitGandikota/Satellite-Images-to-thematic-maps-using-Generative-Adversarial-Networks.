# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 13:27:16 2019

@author: Rohit Gandikota and Radha Krishna
"""

import os
import numpy as np
from osgeo import gdal
#datagen = ImageDataGenerator()
#TASK TO DO.
#THERE ARE TWO IMAGES TO LOAD HERE. 1 IS THE MAIN SAT IMAGE AND THE OTHER IS THE WATER IMAGE.
def load_data(batch_size=1, is_testing=False):
    data_type = "train" if not is_testing else "test"

    data_main='C:\\Users\\user\\Desktop\\Projects\\ImageTileFD\\data_roads\\'
    data_sat='C:\\Users\\user\\Desktop\\Projects\\ImageTileFD\\data_roads\\Data\\'
    data_water='C:\\Users\\user\\Desktop\\Projects\\ImageTileFD\\data_roads\\Labels\\'
    images_water =  os.listdir(data_water)
    images_sat = os.listdir(data_sat)
    
    args = np.intersect1d(images_water, images_sat)
    
    batch_images = np.random.choice(args, size=batch_size)

    sat_data = []
    water_data = []
    for img_path in batch_images:
        sat_img = gdal.Open(data_sat+img_path).ReadAsArray()
        water_img=gdal.Open(data_water+img_path).ReadAsArray()
        water_img[water_img!=water_img]= 0
        water_img[water_img>0] = 1
        sat_img = np.einsum('ijk->jki', sat_img)
        sat_img = (sat_img - sat_img.min()) / (sat_img.max() - sat_img.min())       
        pad = np.zeros((256,256,3))
        pad_w = np.zeros((256,256))
        pad[:220,:220,:]=sat_img
        pad_w[:220,:220]=water_img
#            sat_img = (np.zeros(256,256,3)[:220,:220]=sat_img)
        sat_data.append(pad)
        water_data.append(pad_w)
    water_data = np.array(water_data)
    water_data = np.expand_dims(water_data, axis=-1)
    sat_data = np.array(sat_data)
    
    return water_data,sat_data

def load_batch(batch_size=1, is_testing=False):
    data_type = "train" if not is_testing else "test"
    
    data_main='C:\\Users\\user\\Desktop\\Projects\\ImageTileFD\\data_roads\\'
    data_sat='C:\\Users\\user\\Desktop\\Projects\\ImageTileFD\\data_roads\\Data\\'
    data_water='C:\\Users\\user\\Desktop\\Projects\\ImageTileFD\\data_roads\\Labels\\'
    
    images_water =  os.listdir(data_water)
    images_sat = os.listdir(data_sat)
    
    args = np.intersect1d(images_water, images_sat)
    #batch_images = np.random.choice(os.listdir(data_sat), size=batch_size)
    n_batches = int(len(args) / batch_size)
    for i in range(n_batches-1):

        batch_images = args[i*batch_size:(i+1)*batch_size]
        sat_data = []
        water_data = []
        for img_path in batch_images:
#            print(data_sat+img_path
            sat_img = gdal.Open(data_sat+img_path).ReadAsArray()
            water_img=gdal.Open(data_water+img_path).ReadAsArray()
            water_img[water_img!=water_img]= 0
            water_img[water_img>0] = 1
            sat_img = np.einsum('ijk->jki', sat_img)
            sat_img = (sat_img - sat_img.min()) / (sat_img.max() - sat_img.min())
            pad = np.zeros((256,256,3))
            pad_w = np.zeros((256,256))
            pad[:220,:220,:]=sat_img
            pad_w[:220,:220]=water_img
#            sat_img = (np.zeros(256,256,3)[:220,:220]=sat_img)
            sat_data.append(pad)
            water_data.append(pad_w)
            
        water_data = np.array(water_data)
        water_data = np.expand_dims(water_data, axis=-1)
        
        sat_data = np.array(sat_data)
    
        yield water_data,sat_data

##print(load_data(batch_size=10))
#image_generator=load_batch(batch_size=500)
#water_data, sat_data=next(image_generator)
##