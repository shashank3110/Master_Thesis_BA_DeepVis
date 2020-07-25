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
        #ll=[label_df[label_df['MRI ID']==scan_id.split('.')[0]]['Age'] for scan_id in paths ]
        #print(f'********{ll}**********')
        #print(label_df['MRI ID'])
        '''
        for scan_id in paths:
            age=label_df[label_df['MRI ID']==scan_id.split('.')[0]]['Age']
            gender=label_df[label_df['MRI ID']==scan_id.split('.')[0]]['M/F']
            print(f"age={age},gender={gender}")
            labels.append(age)
            gender.append(gender)
        '''
        labels.extend([label_df[label_df['MRI ID']==scan_id.split('.')[0]]['Age'].to_list()[0] for scan_id in paths ])
        gender.extend([label_df[label_df['MRI ID']==scan_id.split('.')[0]]['M/F'].to_list()[0] for scan_id in paths ])
        cdr.extend([label_df[label_df['MRI ID']==scan_id.split('.')[0]]['CDR'].to_list()[0] for scan_id in paths ])
    #print(labels,gender)
           
    # shuffle(scans)
    
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
        #ll=[label_df[label_df['MRI ID']==scan_id.split('.')[0]]['Age'] for scan_id in paths ]
        #print(f'********{ll}**********')
        #print(label_df['MRI ID'])
        '''
        for scan_id in paths:
            age=label_df[label_df['MRI ID']==scan_id.split('.')[0]]['Age']
            gender=label_df[label_df['MRI ID']==scan_id.split('.')[0]]['M/F']
            print(f"age={age},gender={gender}")
            labels.append(age)
            gender.append(gender)
        '''
        # for scan_id in paths:
        #     cdr_val=label_df[label_df['MRI ID']==scan_id.split('.')[0]]['CDR'].to_list()[0]
        #     if cdr_val == 2:
        #         continue
        #     else:
        #         cdr.append(cdr_val)
        #         labels.append(label_df[label_df['MRI ID']==scan_id.split('.')[0]]['Age'].to_list()[0])
        #         gender.append(label_df[label_df['MRI ID']==scan_id.split('.')[0]]['M/F'].to_list()[0])
        labels.extend([label_df[(label_df['MRI ID']==scan_id.split('.')[0]) & (label_df['CDR']!=2) ]['Age'].to_list()[0] for scan_id in paths ])
        gender.extend([label_df[(label_df['MRI ID']==scan_id.split('.')[0]) & (label_df['CDR']!=2) ]['M/F'].to_list()[0] for scan_id in paths ])
        cdr.extend([label_df[(label_df['MRI ID']==scan_id.split('.')[0]) &  (label_df['CDR']!=2) ]['CDR'].to_list()[0] for scan_id in paths ])
    #print(labels,gender)
           
    # shuffle(scans)
    
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
    train_patients,train_labels,train_gender,_ ,train_cdr= get_scan_from_subjects(data_path,train_ids,data)

    train_cdr = encode_cdrs(train_cdr)

    eval_patients,eval_labels,eval_gender,_,eval_cdr = get_scan_from_subjects(data_path,eval_ids,data)

    eval_cdr = encode_cdrs(eval_cdr)
    # train_patients = [os.path.join(data_path, id) for id in train_ids]
    # eval_patients = [os.path.join(data_path, id) for id in eval_ids]
    # train_labels = [data[data['MRI ID']==id.split('.')[0]]['Age'] for id in train_ids]   
    # eval_labels = [data[data['MRI ID']==id.split('.')[0]]['Age'] for id in eval_ids]  
    # train_gender = [data[data['MRI ID']==id.split('.')[0]]['M/F'] for id in train_ids]  
    # eval_gender = [data[data['MRI ID']==id.split('.')[0]]['M/F'] for id in eval_ids]  
    return train_patients,eval_patients,train_labels,train_gender,train_cdr, eval_labels,eval_gender,eval_cdr

def get_test_files(label_path,data_path):

    data = pd.read_csv(label_path)
    data = data.rename(columns={'MR ID':'MRI ID'})
    data['M/F'] = encode_gender(data)
    
    test_ids = os.listdir(data_path)
    shuffle(test_ids)
    test_patients,test_labels,test_gender,scan_ids,test_cdr = get_scan_from_subjects(data_path,test_ids,data)

    test_cdr = encode_cdrs(test_cdr)
    #test_patients = [os.path.join(data_path, id) for id in test_ids ]
    
    #test_labels = [data[data['MRI ID']==id.split('.')[0]]['Age'].values[0] for id in test_ids]   
    #test_gender = [data[data['MRI ID']==id.split('.')[0]]['M/F'] for id in test_ids]  

    return test_patients,scan_ids, test_labels,test_gender,test_cdr

def encode_gender(data):
    data['M/F'] = pd.Categorical(data['M/F'])
    
    return data['M/F'].cat.codes

def encode_cdrs(test_cdr):
    
    #this wont work as cdr 2 may not be present in one of the train/test/eval sets due to its small count
    # cdr_one_hot = pd.get_dummies(test_cdr)

    # cdr_encoded = cdr_one_hot.values

    # cdr_ohe_dict={0:[1.0,0.0,0.0,0.0],0.5:[0.0,1.0,0.0,0.0],1:[0.0,0.0,1.0,0.0],2:[0.0,0.0,0.0,1.0]}
    cdr_ohe_dict={0:[1.0,0.0,0.0],0.5:[0.0,1.0,0.0],1:[0.0,0.0,1.0]}#if training without cdr2
    cdr_encoded = [cdr_ohe_dict[cdr] for cdr in test_cdr]

    return cdr_encoded
