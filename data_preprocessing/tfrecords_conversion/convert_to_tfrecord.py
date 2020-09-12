
import tensorflow as tf
import numpy as np
import  read_nii
import os
import random
import pandas as pd
from collections import defaultdict
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

# def split_using_subject_id(subject_df,path):
#     '''
#     Split based on subject ids and not scan ids
#     so that multiple scans of a patient are not scattered 
#     across train & test set.

#     '''   
#     df = pd.read_csv(subject_df)
#     length=len(df)
#     subject_list=df['Subject ID'].tolist()
#     mri_count_list = df['MR Sessions'].tolist()
#     split_index = int(train_test_split*length)  
#     train_subjects = subject_list[:split_index]
#     test_subjects = subject_list[split_index:]
#     train_sess_count = mri_count_list[split_index:]
#     test_sess_count = mri_count_list[:split_index]
#     train_session_dict=dict(zip(train_subjects,train_sess_count))
#     test_session_dict=dict(zip(test_subjects,test_sess_count))
#     train_path_list=[]
#     test_path_list=[]

#     for  t in  train_subjects:
#         for c in range(train_session_dict[t]):
#             pat_path=os.path.join(path,t+'_MR'+str(c+1)+'/RAW/smwc1mpr-1.nifti.nii')
#             train_path_list.append(pat_path)

#     for  t in  test_subjects:
#         for c in range(test_session_dict[t]):
#             pat_path=os.path.join(path,t+'_MR'+str(c+1),'/RAW/smwc1mpr-1.nifti.nii')
#             test_path_list.append(pat_path)

#     return train_pat_path_list,test_pat_path_list
def get_pat_path(t,mri_list):
    '''
    returns path for each subject's preprocessed mri scan.
    '''
    path_list = []
    for mri in mri_list:
            #if ('OAS3' in t and t in m) or ('OAS1' in t and t[4:] in m):
                pat_dir = os.path.join(path,mri)
                pat_file = [f for f in os.listdir(pat_dir) if 'smwc' in f]
                pat_path = os.path.join(pat_dir,pat_file[0])
                print(f'pat_path={pat_path}')
                path_list.append(pat_path)

    return path_list
def split_using_subject_id(train_test_split,subject_df,path):
    '''
    Split based on subject ids and not scan ids
    so that multiple scans of a patient are not scattered 
    across train & test set.

    '''   
    df = pd.read_csv(subject_df)
    df_subset = df[['Subject','MR ID']]
    records = df_subset.to_dict('records')
    length=len(df)
    print(len(df['Subject']))
    subject_list=df['Subject'].tolist()
    print(f'subject_list={len(subject_list)}')
    # mri_list = df['MR ID']

    # subject_mri_pairs = list(zip(*(subject_list,mri_list)))
    # random.shuffle(subject_mri_pairs)
    subject_mri_dict=defaultdict(list)
    subject_list=list(set(subject_list))

    #Shuffling entire dataset before splitting into train / test
    random.shuffle(subject_list)

    for r in records:
        subject_mri_dict[r['Subject']].append(r['MR ID'])
    #print(subject_mri_dict)
    #backup same as df_subset = df[['Subject','MR ID']]
    #subject_mri_df = pd.DataFrame(subject_mri_dict)
    #subject_mri_df.to_csv('/no_backups/g009/data/subject_mri_df.csv')


    # for s in subject_list:
    #     for mri in mri_list:

    #         subject_mri_dict[s].append(mri)
    # mri_count_list = df['MR ID'].tolist()

    print(len(subject_list))
    split_index = int(train_test_split*len(subject_list))  
    train_subjects = subject_list[:split_index]
    print(len(train_subjects))
    test_subjects = subject_list[split_index:]
    print(len(test_subjects))
    # train_sess_count = mri_count_list[split_index:]
    # test_sess_count = mri_count_list[:split_index]
    # train_session_dict=dict(zip(train_subjects,train_sess_count))
    # test_session_dict=dict(zip(test_subjects,test_sess_count))
    
    train_path_dict=defaultdict(list)
    test_path_dict=defaultdict(list)
    print(f'********** Train Subject Paths *******************')

    for  t in  train_subjects:
        pat_dir = os.path.join(path,t)
        train_path_list = get_pat_path(t,subject_mri_dict[t])
        train_path_dict[t] = train_path_list
    print(length,split_index,train_test_split,len(train_subjects))
    print(len(test_subjects))
    print(f'********** Test Subject Paths *******************')
    for  t in  test_subjects:        
        test_path_list = get_pat_path(t,subject_mri_dict[t])
        test_path_dict[t] = test_path_list
        # for m in range(mri_list):
        #     if ('OAS3' in t and t in m) or ('OAS1' in t and t[4:] in m):
        #         pat_file = [f for f in os.listdir(pat_dir) if 'smwc' in f]
        #         pat_path = os.path.join(pat_dir,pat_file[0])
        #         print(f'pat_path={pat_path}')
        #         train_path_list.append(pat_path)

    return train_path_dict,test_path_dict,subject_mri_dict

def convert_to_tfr(path_dict,path_tf,subject_mri_dict):
    '''
    Prepares target tf record path and calls the tfrecord conversion func.
    '''
    count=0
    for subject, mri_path_list in path_dict.items() :
        tf_path = os.path.join(path_tf,subject)
        os.system('mkdir '+ tf_path)
        
        for mri_path in mri_path_list:
            image, header, img_shape = read_nii.read(mri_path)
          
            print('Converting to tfrecords...')
            print('Converting :'+ mri_path)

            pat_name = mri_path.split('/')[-1]
            pat_name=pat_name.split('.nii')[0]
            print(f'********pat_name={pat_name}************')
            for id in subject_mri_dict[subject]:
                if id in pat_name and subject.startswith('OAS1'):
                   print(f'********id={id}************')
                   pat_scan = id
                   break
                elif id[-5:] in pat_name and subject.startswith('OAS3'):
                   print(f'********id={id,id[-5:]}************')
                   pat_scan = id
                   break
                else:
                   continue  
            itf_path = os.path.join(tf_path,pat_scan )#pat_name.split('.nii')[0])
            print(f'Target path={itf_path}')
            count+=1
            im2tfrecord(image=image, image_shape = img_shape, path=itf_path + '.tfrecord')
        print(f'************Count={count,len(path_dict)}*******************')   
def convert_all_to_tfr(train_path_dict,train_path_tf,test_path_dict,test_path_tf):
    convert_to_tfr(train_path_dict,train_path_tf)
    convert_to_tfr(test_path_dict,test_path_tf)

def split_using_subject_id_train_cdr0(train_test_split,subject_df,path):
    '''
    Split based on subject ids and not scan ids
    so that multiple scans of a patient are not scattered 
    across train & test set.

    '''   
    df = pd.read_csv(subject_df)
    df_subset = df[['Subject','MR ID','CDR']]
    records = df_subset.to_dict('records')
    length=len(df)

    subject_list=df['Subject'].tolist()
    cdr_list = df['CDR'].to_list()
    # mri_list = df['MR ID']

    # subject_mri_pairs = list(zip(*(subject_list,mri_list)))
    # random.shuffle(subject_mri_pairs)
    subject_mri_dict=defaultdict(list)
    subject_list=list(set(subject_list))

    #Shuffling entire dataset before splitting into train / test
    random.shuffle(subject_list)

    for r in records:
        subject_mri_dict[r['Subject']].append(r['MR ID'])

    #backup same as df_subset = df[['Subject','MR ID']]

    # subject_mri_df = pd.DataFrame(subject_mri_dict)
    # subject_mri_df.to_csv('/no_backups/g009/data/subject_mri_df.csv')

    #############
    # for s in subject_list:
    #     for mri in mri_list:

    #         subject_mri_dict[s].append(mri)
    # mri_count_list = df['MR ID'].tolist()

    
    split_index = int(train_test_split*length)  
    train_subjects = subject_list[:split_index]
    test_subjects = subject_list[split_index:]
    # train_sess_count = mri_count_list[split_index:]
    # test_sess_count = mri_count_list[:split_index]
    # train_session_dict=dict(zip(train_subjects,train_sess_count))
    # test_session_dict=dict(zip(test_subjects,test_sess_count))
    
    train_path_dict=defaultdict(list)
    test_path_dict=defaultdict(list)
    print(f'********** Train Subject Paths *******************')
    #as here we train for only cdr 0 therefore we move non cdr 0 scans to test set.
    for  i,t in  enumerate(train_subjects):
        cdr_list=df_subset[df['Subject']==t]['CDR'].values
        print(f'subject={t},cdrs={cdr_list}')
        cdr_set = set(cdr_list)
        if len(cdr_set) ==1 and 0.0 in cdr_set:
    
            pat_dir = os.path.join(path,t)
            train_path_list = get_pat_path(t,subject_mri_dict[t])
            train_path_dict[t] = train_path_list
        else:
            test_path_list = get_pat_path(t,subject_mri_dict[t])
            test_path_dict[t] = test_path_list
    
    print(f'********** Test Subject Paths *******************')
    for  t in  test_subjects:        
        test_path_list = get_pat_path(t,subject_mri_dict[t])
        test_path_dict[t] = test_path_list

    return train_path_dict,test_path_dict

if __name__ == '__main__':

    #path = '/media/shashanks/Windows/Users/Shashank_S/linux_partition/BA_estimation/OASIS2/OAS2_RAW_PART1/' #.nii files path
    path = '/no_backups/g009/data/OASIS/OASIS1_3_combined/' #.nii files path
    train_path_tf = '/no_backups/g009/data/OASIS/tfrecords_data/training_cdr0/' #path to store tfrecords
    test_path_tf = '/no_backups/g009/data/OASIS/tfrecords_data/testing_all_cdr/'
    if not  os.path.exists(train_path_tf):
       os.makedirs(train_path_tf)
    if not  os.path.exists(test_path_tf):
       os.makedirs(test_path_tf)
    typename = 'nii'
    b_custom = False
    train_test_split = 0.8
    # The below code has to be modified in accordance with the path where .nii files are stores, the naming of the .nii files and so on..
    
    pats = os.listdir(path)
    # len_df={'OASIS1':416,'OASIS2':373,'OASIS3':1098}
    

    ###### old
    # pat_path_list = []
    # for pat in pats:

    #         #pat_path = os.path.join(path, pat+'/RAW/smwc1mpr-1.nifti.nii')
    #         pat_dir = os.path.join(path,pat)
    #         pat_file = [f for f in os.listdir(pat_dir) if 'smwc' in f]
    #         pat_path = os.path.join(pat_dir,pat_file[0])
    #         print(f'pat_path={pat_path}')
    #         #pat_path = os.path.join(path, pat+'/RAW/smwc1mpr-1.nifti.nii')
    #         pat_path_list.append(pat_path)
    
    # path_name_pairs = list(zip(*(pat_path_list,pats)))
    # random.shuffle(path_name_pairs)
    # split_index = int(train_test_split*len(path_name_pairs))
    # train_pat_path_list = path_name_pairs[:split_index]
    # test_pat_path_list = path_name_pairs[split_index:] 
    ######

    ######
    ## comment lines 79 to 91 with this line below to get data split paths
    subject_df='/no_backups/g009/data/oasis1_oasis3_labels.csv'
    train_path_dict,test_path_dict,subject_mri_dict=split_using_subject_id(train_test_split,subject_df,path)
    #convert_all_to_tfr(train_path_dict,train_path_tf,test_path_dict,test_path_tf)
    #all cdr training
    #train_path_dict,test_path_dict=split_using_subject_id(train_test_split,subject_df,path)
    #cdr 0 training
    train_path_dict,test_path_dict=split_using_subject_id_train_cdr0(train_test_split,subject_df,path)
    convert_to_tfr(train_path_dict,train_path_tf,subject_mri_dict)
    convert_to_tfr(test_path_dict,test_path_tf,subject_mri_dict)
    #convert_to_tfr(train_path_dict,test_path_tf)
    # for subject, mri_path_list in train_path_dict.items() :
    #     tf_path = os.path.join(train_path_tf,subject)
    #     os.mkdir(tf_path)
    #     for mri_path in mri_path_list:
    #         image, header, img_shape = read_nii.read(mri_path)
          
    #         print('Converting to tfrecords...')
    #         print('Converting :'+ mri_path)

    #         pat_name = mri_path.split('/')[-1]

    #         tf_path = os.path.join(tf_path, pat_name)
    #         print(f'Target path={tf_path}')
    #         im2tfrecord(image=image, image_shape = img_shape, path=tf_path + '.tfrecord')


    
    
    # for i, mri_path in enumerate(test_path_list):
        
    #     image, header, img_shape = read_nii.read(mri_path)
        
    #     print('Converting to tfrecords...')
    #     print('Converting :'+ mri_path)

    #     pat_name = mri_path.split('/')[-1]

    #     tf_path = os.path.join(test_path_tf, pat_name)
    #     print(f'Target path={tf_path}')
    #     im2tfrecord(image=image, image_shape = img_shape, path=tf_path + '.tfrecord')


    ######old
    # for i, pat in enumerate(train_pat_path_list):
        
    #     image, header, img_shape = read_nii.read(pat[0])
      
    #     print('Converting to tfrecords...')
    #     print('Converting :'+ pat[0])

    #     pat_name = pat[1]#pat[87:-4]
    #     tf_path = os.path.join(train_path_tf, pat_name)
    #     im2tfrecord(image=image, image_shape = img_shape, path=tf_path + '.tfrecord')
    
    # for i, pat in enumerate(test_pat_path_list):
        
    #     image, header, img_shape = read_nii.read(pat[0])
        
    #     print('Converting to tfrecords...')
    #     print('Converting :'+ pat[0])

    #     pat_name = pat[1]#pat[87:-4]
    #     tf_path = os.path.join(test_path_tf, pat_name)
    #     im2tfrecord(image=image, image_shape = img_shape, path=tf_path + '.tfrecord')
