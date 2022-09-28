# -*- coding: utf-8 -*-
"""
Created on Mon May  3 15:27:12 2021

@author: Seda
"""

import os

NUM_PARALLEL_EXEC_UNITS = 4
os.environ['OMP_NUM_THREADS'] = str(NUM_PARALLEL_EXEC_UNITS)

import numpy as np
from PIL import Image
import tensorflow as tf
import numpy.ma as ma
import matplotlib.pyplot as plt


def create_change_label_pixel(idx_list, label, row, col, run):
   
    s_row = row - (row%5)
    s_col = col - (col%5)
    
    a =np.zeros((row,col), dtype=np.uint8)

    shuffled_idx_label_0 = np.empty(idx_list.shape, dtype=np.uint8)
    
    shuffled_idx_label = list(shuffled_idx_label_0)
    for old_index, new_index in enumerate(idx_list):
        shuffled_idx_label[old_index] = label[new_index]
    
    if run == "binary":
        for i in range(s_row-5):
            for j in range(s_col-5):
                prd = shuffled_idx_label[i*(s_col-5)+j]
                if (np.sum(np.all(np.equal(prd, np.array([1, 0])), axis=0))==1):
                    a[i,j] = 0
                elif np.sum(np.all(np.equal(prd, np.array([0, 1])), axis=0))==1:
                    a[i,j] = 1
    elif run == "multiclass":
        for i in range(s_row-5):
            for j in range(s_col-5):
                prd = shuffled_idx_label[i*(s_col-5)+j]
                if (np.sum(np.all(np.equal(prd, np.array([1, 0, 0, 0])), axis=0))==1):
                    a[i,j] = 0
                elif np.sum(np.all(np.equal(prd, np.array([0, 1, 0, 0])), axis=0))==1:
                    a[i,j] = 1
                elif np.sum(np.all(np.equal(prd, np.array([0, 0, 1, 0])), axis=0))==1:
                    a[i,j] = 2
                elif np.sum(np.all(np.equal(prd, np.array([0, 0, 0, 1])), axis=0))==1:
                    a[i,j] = 3

    return a
    

def plot_images_with_mask (image,mask):
    Image_mask = ma.masked_array(mask > 0,image)
    
    plt.imshow(image, cmap='rgb') # I would add interpolation='none'
    
    import matplotlib.pyplot as plt
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(image, 'rgb', interpolation='none')
    plt.subplot(1,2,2)
    plt.imshow(image, 'rgb', interpolation='none')
    plt.imshow(Image_mask, 'jet', interpolation='none', alpha=0.7)
    plt.show()
    