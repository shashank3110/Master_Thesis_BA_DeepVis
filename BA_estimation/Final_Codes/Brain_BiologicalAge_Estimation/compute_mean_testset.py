import copy
import os
import torch
import numpy as np
from collections import defaultdict


exp='exp_ba'
healthy_path = '/content/drive/My Drive/BA_Estimation/models/{0}/Shuffled_labels_evalhealthy.csv'.format(exp)
ad_path = '/content/drive/My Drive/BA_Estimation/models/{0}/Shuffled_labels_evalAD.csv'.format(exp)
healthy_path = '/content/drive/My Drive/BA_Estimation/models/{0}/testset.csv'.format(exp)
ad_path = '/content/drive/My Drive/BA_Estimation/models/{0}/outlier.csv'.format(exp)
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

label_path= cf['Paths']['labels']
data_path= cf['Paths']['test_tfrecord']

test_patients,scan_ids, test_labels,test_gender,test_cdr = get_test_files(label_path,data_path,debug_mode_subject=sub,selected_scans=scans)
tfr=tf.data.TFRecordDataset(test_patients)
img_tf=tfr.map(map_func=lambda a:parse_function_image(a))
for i,im in enumerate(img_tf):
    print(i+1,im.shape)
    img=im.numpy()
    gender_cdr_full_dict[test_gender[i]][test_cdr[i]].append(img)



slice_id = 48
for g in gender_cdr_full_dict.keys():
          print(f'Gender = {gender_dict[g]}')
          for cdr,_ in gender_cdr_full_dict[g].items():
              print(f'CDR = {cdr}')
              full_imgs = np.array(gender_cdr_full_dict[g][cdr])
              full_mean = np.mean(full_imgs,axis=0)
              print(full_imgs.shape,full_mean.shape)
              matpath =  '/content/drive/My Drive/BA_Estimation/means/{0}/'.format(exp)
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


