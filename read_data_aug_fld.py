# -*- coding: utf-8 -*-
"""
Created on Fri May 14 10:49:22 2021

@author: Seda
"""

import os
import cv2

import numpy as np
import scipy.io
from PIL import Image
import tensorflow as tf

NUM_PARALLEL_EXEC_UNITS = 4
os.environ['OMP_NUM_THREADS'] = str(NUM_PARALLEL_EXEC_UNITS)

major_version = int(tf.__version__.split(".")[0])

def shuffle_in_unison(a, b):
    assert len(a) == len(b) #== len(c)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
   # shuffled_c = np.empty(c.shape, dtype=c.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
        #shuffled_c[new_index] = c[old_index]
    return shuffled_a, shuffled_b, permutation

def shuffle_in_unison_sep(a, b, c):
    assert len(a) == len(b) == len(c)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    shuffled_c = np.empty(c.shape, dtype=c.dtype)
    permutation = np.random.permutation(len(a))
    
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
        shuffled_c[new_index] = c[old_index]
    return shuffled_a, shuffled_b, shuffled_c, permutation

def shuffle_in_unison_sep_test(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b, permutation


def create_train_set(images1, images2, labels):
   
    images1 = np.asarray(images1)
    images2 = np.asarray(images2)
    labels = np.asarray(labels)

    regions = []
    region_labels = []

    for i in range(len(images1)):
        for j in range(396):
            for k in range(396):
                row = j + 2
                col = k + 2
                region1 = images1[i, row-2:row+3, col-2:col+3, :]
                assert region1.shape == (5, 5, 3), "Should be (5, 5, 3), was {}".format(region1.shape)

                region2 = images2[i, row - 2:row + 3, col - 2:col + 3, :]
                assert region2.shape == (5, 5, 3), "Should be (5, 5, 3), was {}".format(region2.shape)

                label = labels[i, row - 2:row + 3, col - 2:col + 3]
                assert label.shape == (5, 5), "Should be (5, 5), was {}".format(label.shape)

                region3=np.concatenate((region1,region2))
                regions.append(region3)
                region_labels.append(label)

    regions = np.asarray(regions)
    region_labels = np.asarray(region_labels)

    regions,  region_labels = shuffle_in_unison(regions, region_labels)

    return regions, region_labels

def create_train_set_seperate(images1, images2, labels, channels):
    images1 = np.asarray(images1)
    images1 = images1.astype('float32')
    # normalize to the range 0-1
    images1 /= 255.0
    
    images2 = np.asarray(images2)
    images2 = images2.astype('float32')
    # normalize to the range 0-1
    images2 /= 255.0
    
    labels = np.asarray(labels)
    labels = labels.astype('float32')

    regions1 = []
    regions2 = []
    region_labels = []

    s,t,u=images1.shape
    s = s-s%5
    t = t-t%5

   
    for j in range(s-5):#s_row):
        for k in range(t-5):#s_col):
            row = j + 2
            col = k + 2
            region1 = images1[row-2:row+3, col-2:col+3, :]
            assert region1.shape == (5, 5, channels), "Should be (5, 5, "+str(channels)+"), was {}".format(region1.shape)

            region2 = images2[ row - 2:row + 3, col - 2:col + 3, :]
            assert region2.shape == (5, 5, channels), "Should be (5, 5, "+str(channels)+"), was {}".format(region2.shape)

            label_reg_1 = labels[row - 2:row + 3, col - 2:col + 3]
            
            if label_reg_1.shape == (5, 5, 3):
                label_reg = label_reg_1[:,:,1]
            else:
                label_reg = label_reg_1   
            assert label_reg.shape == (5, 5), "Should be (5, 5), was {}".format(label_reg.shape)
                
            if (label_reg[2,2]==1 or label_reg[2,2]==255):#np.sum(label_reg)>=13):
                label = 1
                
            elif (label_reg[2,2]==2):
                label = 2
                
            elif (label_reg[2,2]==3):
                label = 3
                
            else:
                label = 0
                    
            regions1.append(region1)
            regions2.append(region2)
            region_labels.append(label)

    regions1 = np.asarray(regions1)
    regions2 = np.asarray(regions2)
    region_labels = np.asarray(region_labels)

    regions1, regions2, region_labels, idx_list = shuffle_in_unison_sep(regions1, regions2, region_labels)

    return regions1, regions2, region_labels, idx_list


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

def create_train_set_from_folder(folder,test_im,channels,run,time1,time2):
    #images = load_images_from_folder(folder)

    regions1 = []
    regions2 = []
    region_labels = []
    
    org_inv = ['original']
    
    for filename in [d for d in os.listdir(folder) if str(d).isdigit()]:
        if test_im!=filename:
            for org in org_inv:
                
                if channels == 3:
                    image1 =cv2.imread(
                        os.path.join(
                            folder,
                            filename,
                            "R" + str(filename) + "_original_"+time1+".tif",
                        )
                    )
                    images1 = np.asarray(image1)
                    images1 = images1.astype('float32')
                    images1 /= 255.0
                    image2 = cv2.imread(
                        os.path.join(
                            folder,
                            filename,
                            "R" + str(filename) + "_original_"+time2+".tif",
                        )
                    )
                    images2 = np.asarray(image2)
                    images2 = images2.astype('float32')
                    images2 /= 255.0
                    
                elif channels in [6, 10]:
                    image1 = scipy.io.loadmat(
                        os.path.join(
                            folder,
                            filename,
                            "R"
                            + str(filename)
                            + "_original_"+time1+"_ch"
                            + str(channels)
                            + ".mat",
                        )
                    )
                    image2 = scipy.io.loadmat(
                        os.path.join(
                            folder,
                            filename,
                            "R"
                            + str(filename)
                            + "_original_"+time2+"_ch"
                            + str(channels)
                            + ".mat",
                        )
                    )
                    images1 = image1["new"]
                    images2 = image2["new"]
                
                
                if run == "binary":
                    label_1 = Image.open(os.path.join(folder,filename,'R'+filename+'_'+org+'_Binary_change_thr.png'))
                    label_1 = np.expand_dims(np.asarray(label_1), axis=0)
                    labels_1 = np.asarray(label_1)
                    # convert from integers to floats
                    labels_1 = labels_1.astype('float32')
                    # normalize to the range 0-1
                    labels_1 /= 255.0
                elif run == "multiclass":
                    label_1 = Image.open(os.path.join(folder,filename,'R'+filename+'_'+org+'_four_change_thr.png'))
                    label_1 = np.expand_dims(np.asarray(label_1), axis=0)
                    labels_1 = np.asarray(label_1)
                    # convert from integers to floats
                    labels_1 = labels_1.astype('float32')
                
               
                s,t,u=images1.shape
                s = s-s%5
                t = t-t%5
                
                
                #for i in range(len(images1)):
                for j in range(s-5):#s_row):
                    for k in range(t-5):#s_col):
                        row = j + 2
                        col = k + 2
                        region1 = images1[row-2:row+3, col-2:col+3, :]
                        assert region1.shape == (5, 5, channels), "Should be (5, 5, "+channels+"), was {}".format(region1.shape)
            
                        region2 = images2[row - 2:row + 3, col - 2:col + 3, :]
                        assert region2.shape == (5, 5, channels), "Should be (5, 5, "+channels+"), was {}".format(region2.shape)
                        
                        label_reg_1 = labels_1[0,row - 2:row + 3, col - 2:col + 3]
                        assert label_reg_1.shape == (5, 5), "Should be (5, 5), was {}".format(label_reg_1.shape)
                        
                        if label_reg_1[2,2]==1 or label_reg_1[2,2]==2 or label_reg_1[2,2]==3:
              
                            rotated1_90 = np.rot90(region1)
                            rotated1_180 = np.rot90(rotated1_90)
                            rotated1_270 = np.rot90(rotated1_180)
                                
                            rotated2_90 = np.rot90(region2)
                            rotated2_180 = np.rot90(rotated2_90)
                            rotated2_270 = np.rot90(rotated2_180)
                                
                            flipped_r1 = np.fliplr(region1)
                            flipped_r1_90 = np.fliplr(rotated1_90)
                            flipped_r1_180 = np.fliplr(rotated1_180)
                            flipped_r1_270 = np.fliplr(rotated1_270)
                               
                            flipped_r2 = np.fliplr(region2)
                            flipped_r2_90 = np.fliplr(rotated2_90)
                            flipped_r2_180 = np.fliplr(rotated2_180)
                            flipped_r2_270 = np.fliplr(rotated2_270)
                                
                            regions1.append(region1)
                            regions1.append(rotated1_90)
                            regions1.append(rotated1_180)
                            regions1.append(rotated1_270)
                            regions1.append(flipped_r1)
                            regions1.append(flipped_r1_90)
                            regions1.append(flipped_r1_180)
                            regions1.append(flipped_r1_270)
                                
                            regions2.append(region2)
                            regions2.append(rotated2_90)
                            regions2.append(rotated2_180)
                            regions2.append(rotated2_270)
                            regions2.append(flipped_r2)
                            regions2.append(flipped_r2_90)
                            regions2.append(flipped_r2_180)
                            regions2.append(flipped_r2_270)
                            
                            if label_reg_1[2,2]==1:#or np.sum(label_reg_2)>=13 or np.sum(label_reg_3)>=13 or np.sum(label_reg_4)>=13 ):
                                label = 1   
                            elif label_reg_1[2,2]==2:
                                label = 2
                            elif label_reg_1[2,2]==3:
                                label = 3
                            
                            region_labels.append(label) #region
                            region_labels.append(label) #because the label for 3 different 
                            region_labels.append(label) #rotation is still 1, append three
                            region_labels.append(label) #more 
                            region_labels.append(label) #flipped90
                            region_labels.append(label) #flipped180
                            region_labels.append(label) #flipped270
                            region_labels.append(label) #more 
                            
                        else: 
                            regions1.append(region1)
                            regions2.append(region2)
                            label = 0
                            region_labels.append(label) #region
        
            print(filename,' Finished!')
    regions1 = np.asarray(regions1)
    regions2 = np.asarray(regions2)
    region_labels = np.asarray(region_labels)

    regions1, regions2, region_labels, idx_list = shuffle_in_unison_sep(regions1, regions2, region_labels)


    return regions1, regions2, region_labels

def create_test_set(image1, image2):
    """
    This function is for execution after training. It takes in two images at the same point in space but at
    different points in time. It then segments them into 5x5 segments WITH NO OVERLAP
    These segments should be fed into the ReCNN, and the output can be stitched together to create the predicted
    change map
    :param image1: A single image from time t
    :param image2: A single image from time t+1
    Images should be synced in space
    """

    """
    NOTE: this is not for the testing set of the ReCNN. I would recommend doing a train/val/test split on the data
    produced by the function create_train_set. This is for after the model is finalized, and we have new sets of images
    to detect change in.
    The difference is that this only produces non-overlapping segments, whereas the other function creates
    overlapping segments which are better for training and model evaluation
    """
    assert image1.shape == image2.shape #== (400, 400, 3)

    regions = []
    #regions2 = []
    
    s,t,u=image1.shape
    s_row = int((s-(s%5))/5);
    s_col = int((t-(t%5))/5);

    for i in range(s_row):
        for j in range(s_col):
            row = (i * 5) + 2
            col = (j * 5) + 2

            region1 = image1[row-2:row+3, col-2:col+3, :]
            region2 = image2[row-2:row+3, col-2:col+3, :]
           
            if region1.shape == region2.shape == (5, 5, 3):
                region3=np.concatenate((region1,region2))
                regions.append(region3)
            else:
                print('i {} j{}'.format(i, j))

    regions = np.asarray(regions)
    #regions2 = np.asarray(regions2)

    assert regions.shape == (s_row*s_col, 10, 5, 3) #regions2.shape == (80*80, 5, 5, 3)

    return regions #1, regions2


def create_test_set_seperate(images1, images2):
    images1 = np.asarray(images1)
    images2 = np.asarray(images2)

    regions1 = []
    regions2 = []

    r,s,t,u=images1.shape

    for i in range(len(images1)):
        for j in range(s-4):#s_row):
            for k in range(t-4):#s_col):
                row = j + 2
                col = k + 2
                region1 = images1[i, row-2:row+3, col-2:col+3, :]
                assert region1.shape == (5, 5, 3), "Should be (5, 5, 3), was {}".format(region1.shape)

                region2 = images2[i, row - 2:row + 3, col - 2:col + 3, :]
                assert region2.shape == (5, 5, 3), "Should be (5, 5, 3), was {}".format(region2.shape)

                regions1.append(region1)
                regions2.append(region2)

    regions1 = np.asarray(regions1)
    regions2 = np.asarray(regions2)

    regions1, regions2, idx_list = shuffle_in_unison_sep_test(regions1, regions2)

    return regions1, regions2, idx_list




