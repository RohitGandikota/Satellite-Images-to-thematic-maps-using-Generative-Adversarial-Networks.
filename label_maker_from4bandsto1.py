# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:33:06 2019

@author: Rohit Gandikota (NR02440)
"""

from osgeo import gdal
import os
from PIL import Image

file_path = 'C:\\Users\\user\\Desktop\\Projects\\ImageTileFD\\data\\image_tiles_1'
out_path = 'C:\\Users\\user\\Desktop\\Projects\\ImageTileFD\\data\\labels\\'
files = os.listdir(file_path)
for file in files:
    try:
        img = gdal.Open('C:\\Users\\user\\Desktop\\Projects\\ImageTileFD\\data\\image_tiles_1\\' + file)
        img = img.ReadAsArray()
        img = img [:3, :]
        img = np.sum(img,axis=0)
        img[img>0] = 1
        result = Image.fromarray((img).astype(np.uint8))
        result.save(out_path+file)
    except Exception as e:
        pass
