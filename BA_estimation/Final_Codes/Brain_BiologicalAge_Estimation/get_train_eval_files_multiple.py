'''
Get train and test scans along with gender and labels.
'''

import os
import pandas as pd
from random import shuffle

def get_scan_from_subjects(data_path,subject_ids,label_df):
    scans=[]
    labels=[]
    gender=[]
    cdr=[]
    ids=[]
    for subject in subject_ids :
        path=os.path.join(data_path,subject)
        paths=os.listdir(path)
        ids.extend([scan_id.split('.')[0] for scan_id in paths])
        scans.extend([ os.path.join(path,scan_path) for scan_path in paths])
 
        labels.extend([label_df[label_df['MRI ID']==scan_id.split('.')[0]]['Age'].to_list()[0] for scan_id in paths ])
        gender.extend([label_df[label_df['MRI ID']==scan_id.split('.')[0]]['M/F'].to_list()[0] for scan_id in paths ])
        cdr.extend([label_df[label_df['MRI ID']==scan_id.split('.')[0]]['CDR'].to_list()[0] for scan_id in paths ])

    
    return scans,labels,gender,ids,cdr
        
def get_scan_from_subjects_without_cdr2(data_path,subject_ids,label_df):
    scans=[]
    labels=[]
    gender=[]
    cdr=[]
    ids=[]
    for subject in subject_ids :
        path=os.path.join(data_path,subject)
        paths=os.listdir(path)
        ids.extend([scan_id.split('.')[0] for scan_id in paths])
        scans.extend([ os.path.join(path,scan_path) for scan_path in paths])
     
        labels.extend([label_df[(label_df['MRI ID']==scan_id.split('.')[0]) & (label_df['CDR']!=2) ]['Age'].to_list()[0] for scan_id in paths ])
        gender.extend([label_df[(label_df['MRI ID']==scan_id.split('.')[0]) & (label_df['CDR']!=2) ]['M/F'].to_list()[0] for scan_id in paths ])
        cdr.extend([label_df[(label_df['MRI ID']==scan_id.split('.')[0]) &  (label_df['CDR']!=2) ]['CDR'].to_list()[0] for scan_id in paths ])

    
    return scans,labels,gender,ids,cdr




def prepare_train_eval_files(label_path,data_path, train_eval_ratio):

    
    data = pd.read_csv(label_path,engine='python')
    data = data.rename(columns={'MR ID':'MRI ID'})
    print(f"Before gender encoding={data['M/F']}")
    data['M/F'] = encode_gender(data)
    print(f"AFter gender encoding={data['M/F']}")
    ids = os.listdir(data_path)
    shuffle(ids)
    split = int(train_eval_ratio*len(ids))
    train_ids = ids[:split]
    eval_ids = ids[split:]
    train_patients,train_labels,train_gender,train_scan_ids,train_cdr= get_scan_from_subjects(data_path,train_ids,data)

    train_cdr = encode_cdrs(train_cdr)

    eval_patients,eval_labels,eval_gender,eval_scan_ids,eval_cdr = get_scan_from_subjects(data_path,eval_ids,data)

    eval_cdr = encode_cdrs(eval_cdr)

    return train_patients,train_labels,train_gender,train_cdr,train_scan_ids,train_ids,eval_patients, eval_labels,eval_gender,eval_cdr,eval_scan_ids,eval_ids

def get_test_files(label_path,data_path):

    data = pd.read_csv(label_path)
    data = data.rename(columns={'MR ID':'MRI ID'})
    data['M/F'] = encode_gender(data)
    
    test_ids = os.listdir(data_path)
    shuffle(test_ids)
    test_patients,test_labels,test_gender,scan_ids,test_cdr = get_scan_from_subjects(data_path,test_ids,data)

    test_cdr = encode_cdrs(test_cdr)


    return test_patients,scan_ids, test_labels,test_gender,test_cdr

def encode_gender(data):
    data['M/F'] = pd.Categorical(data['M/F'])
    
    return data['M/F'].cat.codes

def encode_cdrs(test_cdr):
    

    cdr_ohe_dict={0:[1.0,0.0,0.0,0.0],0.5:[0.0,1.0,0.0,0.0],1:[0.0,0.0,1.0,0.0],2:[0.0,0.0,0.0,1.0]}
    #cdr_ohe_dict={0:[1.0,0.0,0.0],0.5:[0.0,1.0,0.0],1:[0.0,0.0,1.0]}#if training without cdr2
    cdr_encoded = [cdr_ohe_dict[cdr] for cdr in test_cdr]

    return cdr_encoded
