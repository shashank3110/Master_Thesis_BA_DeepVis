import copy
import os
import torch
import numpy as np
from collections import defaultdict
import warnings
import os
import argparse

from random import shuffle
import tensorflow as tf
import math
import numpy as np
import torch
import torchvision
import torch.utils.data
from torch.utils.data import DataLoader,TensorDataset
import logging
from keras.layers import Input
from keras.layers.merge import concatenate
import torch.nn as nn
from torch.nn import BatchNorm3d,Conv3d,ReLU,MaxPool3d,Linear,AdaptiveAvgPool3d,Flatten,Softmax
import torch.nn.functional as F

from datetime import datetime
from torch.utils import data
import time


import skimage.io as sio
import os
import shutil
import pandas as pd
from random import shuffle

from skimage.transform import resize
import skimage.io as sio
from scipy.io import savemat,loadmat
import cv2

import mask
import draw
import norm
import misc

from torchvision import models

from random import shuffle
from torchvision.utils import make_grid, save_image

import pandas as pd
from gradcam.utils import visualize_cam
from gradcam import GradCAMpp, GradCAM
from matplotlib import pyplot as plt
"""#**Read Nii images code**"""

import os





# """#**Saliency Map section begins**"""



def get_selected_scan_from_subjects(data_path,subject_ids,label_df,selected_scans):
    scans=[]
    labels=[]
    gender=[]
    cdr=[]
    ids=[]
    subject_ids = set(subject_ids)
    dpaths = os.listdir(data_path)
    path1 = os.path.join(data_path,dpaths[0])
    path2 = os.path.join(data_path,dpaths[1])
    list1 = os.listdir(path1)
    list2 = os.listdir(path2)
    for subject in subject_ids :
    	if subject in list1:
    		path=os.path.join(path1,subject)
    	elif subject in list2:
        	path=os.path.join(path2,subject)
        paths=os.listdir(path)
        ids.extend([scan_id.split('.')[0] for scan_id in paths  if scan_id.split('/')[-1].split('.')[0] in selected_scans ])
        scans.extend([ os.path.join(path,scan_id) for scan_id in paths   if scan_id.split('/')[-1].split('.')[0] in selected_scans ])
        
    
        labels.extend([label_df[label_df['MRI ID']==scan_id.split('.')[0]]['Age'].to_list()[0] for scan_id in paths   if scan_id.split('/')[-1].split('.')[0] in selected_scans ])
        gender.extend([label_df[label_df['MRI ID']==scan_id.split('.')[0]]['M/F'].to_list()[0] for scan_id in paths   if scan_id.split('/')[-1].split('.')[0] in selected_scans])
        cdr.extend([label_df[label_df['MRI ID']==scan_id.split('.')[0]]['CDR'].to_list()[0] for scan_id in paths  if scan_id.split('/')[-1].split('.')[0] in selected_scans])

    return scans,labels,gender,ids,cdr

def get_scan_from_subjects(data_path,subject_ids,label_df):
    scans=[]
    labels=[]
    gender=[]
    cdr=[]
    ids=[]
    subject_ids = set(subject_ids)
    for subject in subject_ids :
        path=os.path.join(data_path,subject)
        paths=os.listdir(path)
        ids.extend([scan_id.split('.')[0] for scan_id in paths ])
        scans.extend([ os.path.join(path,scan_path) for scan_path in paths  ])
 
        labels.extend([label_df[label_df['MRI ID']==scan_id.split('.')[0]]['Age'].to_list()[0] for scan_id in paths   ])
        gender.extend([label_df[label_df['MRI ID']==scan_id.split('.')[0]]['M/F'].to_list()[0] for scan_id in paths  ])
        cdr.extend([label_df[label_df['MRI ID']==scan_id.split('.')[0]]['CDR'].to_list()[0] for scan_id in paths ])
    #print(labels,gender)
           
    # shuffle(scans)
    
    return scans,labels,gender,ids,cdr


def get_test_files(label_path,data_path,debug_mode_subject=None,selected_scans=[]):

    data = pd.read_csv(label_path)
    data = data.rename(columns={'MR ID':'MRI ID'})
    data['M/F'] = encode_gender(data)
    if debug_mode_subject is None:

      test_ids = os.listdir(data_path)
    else:
      test_ids=debug_mode_subject
    
    shuffle(test_ids)
    if len(selected_scans)>0:
      test_patients,test_labels,test_gender,scan_ids,test_cdr = get_selected_scan_from_subjects(data_path,test_ids,data,selected_scans)
    else:
      test_patients,test_labels,test_gender,scan_ids,test_cdr = get_scan_from_subjects(data_path,test_ids,data)
   
    return test_patients,scan_ids, test_labels,test_gender,test_cdr
   
def encode_gender(data):
    data['M/F'] = pd.Categorical(data['M/F'])
    
    return data['M/F'].cat.codes

def parse_function_image(example_proto):

    features = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'image_shape': tf.io.FixedLenFeature([], tf.string)
    }

    content = tf.io.parse_single_example(example_proto, features=features)

    content['image_shape'] = tf.io.decode_raw(content['image_shape'], tf.int32)
    content['image'] = tf.io.decode_raw(content['image'], tf.float32)
    content['image'] = tf.reshape(content['image'], content['image_shape'])

    return content['image']

exp='exp_ba'
# healthy_path = '../../csv_data/Shuffled_labels_evalhealthy.csv'
# ad_path = '../../csv_data/Shuffled_labels_evalAD.csv'
healthy_path = '../../csv_data/exp_ba/testset.csv'
ad_path = '../../csv_data/exp_ba/outlier.csv'
hdf = pd.read_csv(healthy_path)
adf = pd.read_csv(ad_path)
print(hdf.columns)

sub =  hdf['patient_id'].values.tolist()
sub.extend(adf['patient_id'].values.tolist())
sub= list(set(sub))
scans= copy.deepcopy(sub)
for i,s in enumerate(sub) :
  if s.startswith('OAS1'):
    s= s[:9] #OAS1_0123_MR1 take first 9 characters
    sub[i] = s.replace('_','')
  elif s.startswith('OAS3'): #OAS31098_MR_d7178 #take just subject id
    sub[i] = s.split('_')[0]

gender_dict={0:'Female',1:'Male'}
gender_cdr_chunk_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
gender_cdr_full_dict = defaultdict(lambda: defaultdict(list))
dt_string = datetime.now().strftime('%d-%m-%Y-%H-%M')+'mean_scans_smoe_maps_blockend_scale_endlayers_equal_weights'

label_path= '../../csv_data/oasis1_oasis3_labels.csv'
data_path= ['../../tfrecords_data/OASIS1_3_combined/tfrecords_data_cdr0_training/testing_all_cdr','../../tfrecords_data/OASIS1_3_combined/tfrecords_data_cdr0_training/training_cdr0'
test_patients,scan_ids, test_labels,test_gender,test_cdr = get_test_files(label_path,data_path,debug_mode_subject=sub,selected_scans=scans)
tfr=tf.data.TFRecordDataset(test_patients)
img_tf=tfr.map(map_func=lambda a:parse_function_image(a))
for i,im in enumerate(img_tf):
    print(i+1,im.shape)
    img=im.numpy()
    gender_cdr_full_dict[test_gender[i]][test_cdr[i]].append(img)



slice_id = 9
for g in gender_cdr_full_dict.keys():
          print(f'Gender = {gender_dict[g]}')
          for cdr,_ in gender_cdr_full_dict[g].items():
              print(f'CDR = {cdr}')
              full_imgs = np.array(gender_cdr_full_dict[g][cdr])
              full_mean = np.mean(full_imgs,axis=0)
              print(full_imgs.shape,full_mean.shape)
              matpath =  '../../means/{0}/'.format(exp)
              if not os.path.exists(matpath):
                os.makedirs(matpath)
              savemat(matpath+'full_img_mean_cdr'+str(cdr)+'_'+gender_dict[g]+'.mat',{'data': full_mean,'shape':full_mean.shape})
              
              print('********************************')
              img_a = full_imgs[:,:,:,48:54]
              img_a = np.expand_dims(img_a,-1)
              slice_mean = np.mean(img_a,axis=0)
              savemat(matpath+'slice'+str(slice_id)+'axial_img_mean_cdr'+str(cdr)+'_'+gender_dict[g]+'.mat',{'data': full_mean,'shape':full_mean.shape})
              print(slice_mean.shape)
              
              img_s = np.mean(full_imgs[:,48:54,:,:],axis=0)
              img_s=np.expand_dims(img_s,0)
              slice_mean = torch.from_numpy(img_s).permute(2,1,0).numpy()
              print(slice_mean.shape)
              savemat(matpath+'slice'+str(slice_id)+'sagittal_img_mean_cdr'+str(cdr)+'_'+gender_dict[g]+'.mat',{'data': full_mean,'shape':full_mean.shape})
              
              img_c = np.mean(full_imgs[:,:,48:54,:],axis=0)
              # img_c=np.expand_dims(img_c,0)
              img_c= torch.from_numpy(img_c).permute(1,0)

              img_c=img_c.unsqueeze(-1)
              img_c=img_c.unsqueeze(0)
              img_c = torch.nn.functional.upsample(img_c.unsqueeze(0), size=(121,145,1), mode='nearest') 
              slice_mean = img_c[0,0,:,:,:].numpy()
              print(slice_mean.shape)
              savemat(matpath+'slice'+str(slice_id)+'coronal_img_mean_cdr'+str(cdr)+'_'+gender_dict[g]+'.mat',{'data': full_mean,'shape':full_mean.shape})


