# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 17:21:12 2019

@author: user
"""
osm_wms='http://layered.wms.geofabrik.de/std/demo_key?'
from owslib.wms import WebMapService
URL = osm_wms
wms = WebMapService(URL, version='1.1.1')

OUTPUT_DIRECTORY = './data/image_tiles_1/'
#Hyderabad=17.42723, 78.48594
#kapra=17.48422, 78.56439
x_min = 78.48594
y_min = 17.4273
dx, dy = 0.01, 0.01
no_tiles_x = 100
no_tiles_y = 100
total_no_tiles = no_tiles_x * no_tiles_y

x_max = x_min + no_tiles_x * dx
y_max = y_min + no_tiles_y * dy
BOUNDING_BOX = [x_min, y_min, x_max, y_max]

for ii in range(0,no_tiles_x):
    print(ii)
    for jj in range(0,no_tiles_y):
        ll_x_ = x_min + ii*dx
        ll_y_ = y_min + jj*dy
        bbox = (ll_x_, ll_y_, ll_x_ + dx, ll_y_ + dy) 
        img = wms.getmap(layers=['s2cloudless-2018'], srs='EPSG:4326', bbox=bbox, size=(256, 512), format='image/png', transparent=True)
        filename = "im_{}_{}_{}_{}.png".format(str(bbox[0]).replace('.','_'), str(bbox[1]).replace('.','_'), str(bbox[2]).replace('.','_'),str(bbox[3]).replace('.','_'))
        out = open(OUTPUT_DIRECTORY + filename, 'wb')
        out.write(img.read())
        out.close()