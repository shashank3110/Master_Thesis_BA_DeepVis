# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 11:51:40 2020

@author: Anish
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 19:17:07 2019

@author: Anish
"""
import tensorflow as tf
import numpy as np
import  read_nii
import os

#tf.enable_eager_execution()

def im2tfrecord(image, image_shape, path):
    """
     Takes the image and saves it to a tfrecord
     using image and image.shape info
    """
    image = np.asarray(image)
    # np.frombuffer(np.array(a.shape).tostring(), dtype=int)  # numpy supports string conversion
    image_shape = np.array(image_shape, dtype=np.int32).tostring()
    image = image.astype(np.float32).tostring()
    #image = image.tostring()
    print(type(image))
    print(type(image_shape))
    # create an example protocol buffer
    def _floats_feature(value):
         return tf.train.Feature(float_list=tf.train.FloatList(value=value.reshape(-1)))
     
    feature = {
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
        'image_shape': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_shape]))
    }
    features = tf.train.Features(feature=feature)
    example = tf.train.Example(features=features)

    with tf.io.TFRecordWriter(str(path)) as writer:
        writer.write(example.SerializeToString())
        
if __name__ == '__main__':

    path = '/media/shashanks/Windows/Users/Shashank_S/linux_partition/BA_estimation/OASIS2/OAS2_RAW_PART1/OAS2_0001_MR1/RAW/'#'D:/4th_semester/Tensorflow_Codes/Biological_Age_Estimation/knee_mri_nii/' #.nii files path
    path_tf = '/media/shashanks/Windows/Users/Shashank_S/linux_partition/BA_estimation/tfrecords_data/'#'D:/4th_semester/Tensorflow_Codes/Biological_Age_Estimation/knee_mri_nii_tfrecords_new/' #path to store tfrecords

    typename = 'nii'
    b_custom = False

    # The below code has to be modified in accordance with the path where .nii files are stores, the naming of the .nii files and so on..
    pats = os.listdir(path)
    pat_path_list = []

    for pat in pats:
	    if pat.startswith('wc1'):

		    pat_path = os.path.join(path, pat)

		    pat_path_list.append(pat_path)

    
    for i, pat in enumerate(pat_path_list):
        
        image, header, img_shape = read_nii.read(pat)
        #print(image)
        print(image.shape)
        #print(header)
        pat_name = pat[72:-4]
#        if image.shape==(320,304,37):
#            print(pat_name)
        #im2tfrecord(image=image, image_shape = img_shape, path=path_tf + pat_name + '.tfrecord')
    for i, pat in enumerate(pat_path_list):
        
        image, header, img_shape = read_nii.read(pat)
      
        print('Converting to tfrecords...')
        print('Converting :'+ pat)

        pat_name = pat.split('/')[-1]#pat[87:-4]
        tf_path = os.path.join(path_tf, pat_name)
        im2tfrecord(image=image, image_shape = img_shape, path=tf_path + '.tfrecord')
