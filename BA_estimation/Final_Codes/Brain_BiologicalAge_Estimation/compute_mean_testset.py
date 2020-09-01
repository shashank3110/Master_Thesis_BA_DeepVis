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


def get_smoe_map(x,relu=False):
  print(f' smoe input shape={x.shape}')
  if relu:
    x=tf.nn.relu(x).numpy()
  print(f'x range={np.amax(x),np.amin(x)}')
  
  m   = np.mean(x,axis=-1)+0.0000001 
  # m   = np.mean(x,axis=0)+0.0000001 #avoid log 0
  
  x   = x + 0.0000001
  # k   = np.log2(m) - np.mean(np.log2(x), axis=0)
  k   = np.log2(m) - np.mean(np.log2(x), axis=-1)
  print(f'log of mean={np.log2(m)}, mean of log={np.mean(np.log2(x), axis=-1)}')
  print(f'k={k}')
  k   = k + 0.0000001
  # k   = np.log10(m) - np.mean(np.log10(x), axis=-1)
  print(np.array_equal(np.zeros(k.shape),k))
  print(f'{x.shape,k.shape,np.amin(k)}')
  print(f'kmax, kmin={np.min(k),np.max(k)}')
  print(f'mean={m}')
  th  = k * m
  print(f'smoe map={th}')
  print(f'smoe output shape={th.shape}')
  return th

def get_std_map(x):
  print(f'before std map shape ={x.shape}')
  m = np.std(x,axis=-1)

  print(f'std map shape ={m.shape}')

  return m

def get_norm(x,const_mean=None,const_std=None):
  # print(f'x shape={x.shape}')
  s0      = x.shape[0]
  s1      = x.shape[1]
  s2      = x.shape[2]

  # x       = np.reshape(x,(1,s0*s1))
  x       = np.reshape(x,(1,s1*s2))
  print(f'get norm func x after reshape={x.shape} ')

  '''
      Compute Mean
  '''
  if const_mean is None:
      m       = np.mean(x,axis=1)
      m       = np.reshape(m,(m.shape[0],1))
  else:
      m       = const_mean

  print(f'get norm func x after mean reshape={m.shape} ') 
  '''
      Compute Standard Deviation
  '''
  if const_std is None:
      s       = np.std(x,axis=1)
      s       = np.reshape(s,(s.shape[0],1))
  else:
      s       = const_std
  
  '''
      The normal cumulative distribution function is used to squash the values from within the range of 0 to 1
  '''

  s=torch.tensor(s)
  x       = 0.5*(1.0 + torch.erf((x-m)/(s*torch.sqrt(torch.tensor(2.0)))))
  print(x.shape)    
  # x       = x.reshape(1,s0,s1)
  x       = x.reshape(1,s1,s2)

  print(f'map after norm={x,x.shape}')
  return x


def combine_sal_maps(smaps,output_size,weights,map_num,resize_mode='bilinear',do_relu=False):
  bn  = smaps[0].shape[0]
  cm  = torch.zeros((bn, 1, output_size[0], output_size[1]), dtype=smaps[0].dtype, device=smaps[0].device)
  ww  = []
  
  '''
      Now get each saliency map and resize it. Then store it and also create a combined saliency map.
  '''
  for i in range(len(smaps)):
      # assert torch.is_tensor(smaps[i]), "Each saliency map must be a Torch Tensor."
      wsz = smaps[i].shape
      w   = np.reshape(smaps[i],(wsz[0], 1, wsz[1], wsz[2]))#smaps[i].reshape(wsz[0], 1, wsz[1], wsz[2])
   
      w   = nn.functional.interpolate(w, size=output_size, mode=resize_mode, align_corners=False) 
      ww.append(w)        # should we weight the raw maps ... hmmm
      
      cm  += (w * weights[i])

  '''
      Finish the combined saliency map to make it a weighted average.
  '''
  weight_sum =sum(weights)
  cm  = cm / weight_sum
  cm  = cm.reshape(bn, output_size[0],output_size[1])
  
  ww  = torch.stack(ww,dim=1)
  ww  = ww.reshape(bn, map_num, output_size[0], output_size[1])
  

  
  return cm, ww

def compute_saliency_tf(base_path,inputs,tf_model):

  # gender=inputs[1]
  # gender=tf.reshape(gender,[1,1])
  img=inputs
  img_chunk= img #tf.convert_to_tensor(img)
  print(img_chunk.shape)
  img_chunk = tf.reshape(img_chunk,[1,121,145,1])
  layers=[layer.name for layer in tf_model.layers]
  outputs=[]

 
  j=0
  for i,l in enumerate(layers):
    if 'activation' in l:
      val=tf_model.get_layer(name=l).output
      print(i,j,l,val.shape)
      j+=1
      outputs.append(val) 
  outputs.append(encoder.output) 
                                        
  test_tf_model=tf.keras.models.Model(tf_model.inputs, outputs)
 
  predictions = test_tf_model(img_chunk)

  # hooks=[predictions[0],predictions[1],predictions[2],predictions[8],predictions[14],predictions[20],predictions[23]\
  #        ,predictions[29],predictions[35],predictions[41],predictions[47],predictions[50],\
  #        predictions[56],predictions[62],predictions[65]]#predictions[:layer_end]
  # hooks= [predictions[0],predictions[2],predictions[17],predictions[47],predictions[62]] #1x1 and 3x3 cnn
  hooks=[predictions[0],predictions[9],predictions[22],predictions[41],predictions[51]] #1x1 cnn


  # choose specific channels / filters
  for x in hooks:
    print('ouput shapes layerwise')
    print(x.shape)


  #smoe
  # sal_maps       = [ get_norm(get_smoe_map(np.mean(x.numpy()[:,:,:,:],axis=-1))) for x in hooks ]
  sal_maps       = [ get_norm(get_smoe_map(x.numpy())) for x in hooks ]

  #std dev
  # sal_maps       = [ get_norm(get_std_map(np.mean(x.numpy()[:,:,:,:,:],axis=-2))) for x in hooks ]

  # sal_maps       = [ get_norm(get_smoe_map(x.numpy()[:,:,:,:,:])) for x in hooks ]
  for smaps in sal_maps:
    print(smaps.shape)
  
  # all layer scale maps with equal weightage
  weights=np.ones(len(hooks))
 
  map_num=len(hooks)

  f, axarr = plt.subplots(1,1,figsize=(10,10))
  raw= np.mean(img_chunk[0,:,:,:],axis=-1)
  raw= raw/np.max(raw)
  r=axarr.imshow(raw,cmap='jet')
  axarr.set_title('Input image')
  cbar=plt.colorbar(r,fraction=0.01, pad=0.04)
  cbar.set_clim(0,1)
  plt.savefig(base_path+'input_chunk.png')

  csal_maps,sal_maps = combine_sal_maps(sal_maps,output_size=[in_height,in_width],weights=weights,map_num=map_num)
  output_path = base_path +'Map_Combined.png'
  f, axarr = plt.subplots(1,1,figsize=(10,10))
  csal_map=csal_maps[0,:,:].numpy()
  imcs=csal_map/np.max(csal_map)
  im = axarr.imshow(imcs,cmap='jet')
  axarr.set_title('Combined saliency map')
  cbar=plt.colorbar(im,fraction=0.01, pad=0.04)
  cbar.set_clim(0,1)
  plt.savefig(output_path)

  il = [sal_maps[0,i,:,:] for i in range(map_num)] # Put each saliency map into the figure
  il.append(csal_maps[0,:,:])                       # add in the combined map at the end of the figure
  images        = [torch.stack(il, 0)]          
  images        = make_grid(images, nrow=5)
  sal_img=images.unsqueeze(1)
  output_path=base_path +'Sal_Maps.png'
  save_image(sal_img,output_path)

  input_path = output_path
  f, axarr = plt.subplots(1,1,figsize=(10,10))
  im=sio.imread(input_path)
  im=axarr.imshow(np.mean(im,axis=-1)/255, cmap='jet');
  axarr.set_title('layerwise saliency maps')
  cbar=plt.colorbar(im,fraction=0.01, pad=0.04)
  cbar.set_clim(0,1)
  output_path=base_path +'Sal_Maps_jet.png'
  plt.savefig(output_path)


  

  return csal_maps



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
        elif subject in list2
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


