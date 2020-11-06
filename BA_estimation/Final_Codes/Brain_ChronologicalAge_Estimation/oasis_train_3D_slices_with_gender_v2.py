'''
This is the main file for training the network for Chronological Age estimation with age and gender. 
The input data strategy of dividing the volume into volume chunks is employed here. 
The hybrid 3D CNN network is used for training.
'''

import os
import cv2
import sys
import argparse
import shutil
import yaml
import matplotlib.pyplot as plt
import math
import tensorflow as tf
import pandas as pd
from pylab import *
import numpy as np
import keras
from tensorflow.keras import Model
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras import callbacks as cb
from keras.utils import multi_gpu_model
from tensorflow.keras.callbacks import (CSVLogger,\
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    LearningRateScheduler
)

from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
K.set_image_data_format = 'channels_last'
from tensorflow.compat.v1.keras.backend import set_session,clear_session,get_session

import util.generator_3D_volume_slices_age_with_gender as generator
import get_train_eval_files_multiple as get_train_eval_files
import numpy
import gc
from network import Hybrid3DCNN_gender_age_v2,Hybrid3DCNN_gender_age_v2_classification,Hybrid3DCNN_oasis
from datetime import datetime
from skimage.transform import resize


print(tf.__version__)
total_gpus=tf.config.experimental.list_physical_devices('GPU')
print(f'total_gpus={total_gpus}')
gpu=total_gpus[0]
tf.config.experimental.set_visible_devices(gpu,'GPU')

logical_gpus = tf.config.experimental.list_logical_devices('GPU')
print(f'GPUs used = {logical_gpus}')
#Use the below class only if you want to train on several GPUs
class ModelMGPU(Model):
    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
           '''
           Override load and save methods to be used from the serial-model. The
           serial-model holds references to the weights in the multi-gpu model.
           '''
        if 'load' in attrname or 'save' in attrname:
           return getattr(self._smodel, attrname)
        else:

           return super(ModelMGPU, self).__getattribute__(attrname)


def train(cf,exp_name):
    
    #Choose the GPU on which the training should run    
    print(f'GPUs used = {logical_gpus}')


    train_data_path = cf['Paths']['train_tfrecord']
    train_label_path = cf['Paths']['labels']
    train_eval_ratio = cf['Data']['train_val_split']
    batch_size = cf['Training']['batch_size']
    samples = cf['Training']['samples']
    image_shape = cf['Training']['image_shape']
    num_parallel_calls = cf['Training']['num_parallel_calls']
    epoch = cf['Training']['num_epochs']
    print(f"type(epoch)={type(epoch)}")
    

    
    #Divide the entire dataset into training and validation set
    train_patients,eva_patients,train_labels, train_labels_gender, train_cdr,eva_labels, eva_labels_gender,eval_cdr \
         = get_train_eval_files.prepare_train_eval_files(train_label_path,train_data_path, train_eval_ratio)
    

    
    print('Length of training patients is: ', len(eva_labels))
    print('The training patients are: ',train_patients)
    print('Length of validation patients is: ', len(eva_labels)) 
    print('The validation patients are: ',eva_patients)
    print('The ages of train patients are: ',train_labels)
    print('The cdrs of train patients are: ',train_cdr)
    print('The genders of train patients are: ',train_labels_gender)
    print('The ages of validation patients are: ',eva_labels)
    print('The genders of validation patients are: ',eva_labels_gender)
    print('The cdrs of validation patients are: ',eval_cdr)

    #The total count has to be multiplied by the number of samples per patient. In this case it is 20 volume chunks per patient
    count_train= len(train_patients) * samples
    count_validation = len(eva_patients) * samples
    print(f'count_train={count_train},count_validation={count_validation}')
    steps_per_epoch = math.ceil(count_train / batch_size) + 1
    validata_steps = math.ceil(count_validation / batch_size) + 1

    print('Expected training steps: ', steps_per_epoch)
    print('Expected validation_steps: ', validata_steps)
    print('Batch size : ', batch_size)

    print('-' * 75)
    print(' Model\n')

    #Create the model with the shape of the input
    input_size = image_shape + [1]

    print(f"classification flag={cf['Classification']}")
    if cf['Classification'] =='Y' : 
        model,_ =  Hybrid3DCNN_gender_age_v2_classification.createModel(input_size)
        train_labels =  train_cdr #switch labels from age to cdr (regression to classification)
        eva_labels = eval_cdr
    else:
        model,_ = Hybrid3DCNN_oasis.createModel(input_size)  #Hybrid3DCNN_gender_age_v2.createModel(input_size)

    
    #Uncomment the below command in case of using multiple GPUs
    #model = ModelMGPU(model, 2)
    
    #Defining RMSE metric
    def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 

    learning_rate = cf['Training']['learning_rate']
    
    
    
    print('learning rate is:',learning_rate )
    adm = optimizers.Adam(lr=learning_rate)
    if cf['Classification'] =='Y' : 
        model.compile(loss='categorical_crossentropy', optimizer=adm, \
            metrics=[tf.keras.metrics.AUC(),'categorical_accuracy', 'accuracy'])
    else:
        model.compile(loss='mse', optimizer=adm, metrics=['mae', rmse])
    model.summary()
    print(' Model compiled!')
   
    #Defining the callback functions
    def get_callbacks(model_file, logging_file,checkpoint_path,classification_flag):
        callbacks = list()
        print('###Entered callback func ###')
        if classification_flag == 'Y':
            #Save the model
            callbacks.append(cb.ModelCheckpoint(model_file, monitor='val_auc', save_best_only=True, mode='max'))
            print('###step1 in callback func ###')
            # Save ckpt
            print('###step2 in callback func ###')
            callbacks.append(cb.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1))
            #Save the log file
            callbacks.append(CSVLogger(logging_file, append=True))
            print('###step3 in callback func ###')
            #Reduce LR on plateau
            callbacks.append(ReduceLROnPlateau(monitor='val_auc',mode='auto', factor=0.2, patience=12, min_lr=1e-7))
            print('###step4 in callback func ###')
            #Stop training in case of validation error increase
            callbacks.append(EarlyStopping(monitor='val_auc', min_delta=0.005, patience=18, verbose=0,\
             mode='auto', baseline=None, restore_best_weights=False))
            print('###step5 leaving  callback func ###')
        else:
            #Save the model
            callbacks.append(cb.ModelCheckpoint(model_file, monitor='val_mae', save_best_only=True, mode='min'))
            print('###step1 in callback func ###')
            # Save ckpt
            print('###step2 in callback func ###')
            callbacks.append(cb.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1))
            #Save the log file
            callbacks.append(CSVLogger(logging_file, append=True))
            print('###step3 in callback func ###')
            #Reduce LR on plateau
            callbacks.append(ReduceLROnPlateau(monitor='val_mae', factor=0.2, patience=12, min_lr=1e-7))
            print('###step4 in callback func ###')
            #Stop training in case of validation error increase
            callbacks.append(EarlyStopping(monitor='val_mae', min_delta=0.005, patience=18, verbose=0,\
             mode='auto', baseline=None, restore_best_weights=False))
            print('###step5 leaving  callback func ###')
        return callbacks

    print('-' * 75)
    print(' Training...')

    #Use a custom data generator to generate volume chunks from a complete volume of a patient
    


    #For the training patients
    

    train_generator = generator.tfdata_generator_volume_chunks(file_lists=train_patients,
                                                                      label_lists=train_labels,
                                                                      label_gender=train_labels_gender,
                                                                      num_parallel_calls=num_parallel_calls,
                                                                      train_patch_size = image_shape,
                                                                      samples = samples

                                                                      )

    train_generator = generator.batch_and_run(train_generator, batch_size, count_train, case='train')

    #For the validation patients
    val_generator = generator.tfdata_generator_volume_chunks(file_lists=eva_patients,
                                                                    label_lists=eva_labels,
                                                                    label_gender=eva_labels_gender,
                                                                    num_parallel_calls=num_parallel_calls,
                                                                    train_patch_size = image_shape,
                                                                    samples = samples
                                                                    )


    val_generator = generator.batch_and_run(val_generator, batch_size, count_validation, case='valid')

    path_w = cf['Paths']['model'] + "age_net_oasis1_3" + ".hdf5"
    logging_file = cf['Paths']['model'] + "age_net_oasis1_3" + ".txt"
    checkpoint_path = cf['Paths']['model'] + "age_net_oasis1_3" + ".ckpt"
    #Start training the model

    history = model.fit(
          train_generator,
          steps_per_epoch=steps_per_epoch,
          epochs = epoch,
          validation_data=val_generator,
          validation_steps=validata_steps,
          callbacks=get_callbacks(model_file=path_w, logging_file=logging_file,\
            checkpoint_path=checkpoint_path,classification_flag=cf['Classification']))
    print(f'training history={history.history}')
                
#Testing on the test set
def test(cf,exp_name):
    

    batch_size = cf['Training']['batch_size']
    image_shape = cf['Training']['image_shape']

    samples = cf['Training']['samples']
    num_parallel_calls = cf['Training']['num_parallel_calls']
    case = cf['Case']

    def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 

    if cf['Pretrained_Model']['path'] is not None:
        print(' Load pretrained model')
        model = tf.keras.models.load_model(filepath=cf['Pretrained_Model']['path'], compile=False)
        print(' Summary of the model:')
        model.summary()
        
    else:
        
        print('no pretrained  model')

    # get test Patient
    test_data_path = cf['Paths']['test_tfrecord']
    test_label_path = cf['Paths']['labels']
    test_patients,test_ids, test_age,test_gender, test_cdr  = get_train_eval_files.get_test_files(test_label_path,test_data_path)

    print(test_age)
    count_test = len(test_patients) * samples
    test_steps = math.ceil(count_test / batch_size)


    # for cdr classification network
    if cf['Classification'] =='Y' : 
        
        test_generator = generator.tfdata_generator_volume_chunks(file_lists=test_patients, label_lists=test_cdr,
    label_gender=test_gender,
                                                              num_parallel_calls=num_parallel_calls,
                                                                          train_patch_size = image_shape,
                                                                          samples = samples
                                                              )
    else:
        

        test_generator = generator.tfdata_generator_volume_chunks(file_lists=test_patients, label_lists=test_age,
        label_gender=test_gender,
                                                                  num_parallel_calls=num_parallel_calls,
                                                                              train_patch_size = image_shape,
                                                                              samples = samples
                                                                  )

    test_generator = generator.batch_and_run(test_generator, batch_size,
                                                 count_test, case=case)

    prediction = model.predict(test_generator, steps=test_steps, verbose=1)
    print(len(test_age))
    print(len(prediction))
    print(f'All predictions={prediction}')
    


    total_patient_name = [ id.split('.')[0] for id in test_ids]
    num_patients = len(test_patients)


    print(f'total_patient_name={total_patient_name}')
    print(f'test_ids={test_ids}')
    print(f'num_patients={num_patients}')

                 

    # convert predicted age to float
    total_predicted_age_regresion = []
    maximum_prediction_list = []
    minimum_prediction_list = []
    minimum_prediction_diff_list = []
    maximum_prediction_diff_list = []
    slice_max = []
    slice_min = []
    cdr_ohe_dict={0:[1.0,0.0,0.0,0.0],0.5:[0.0,1.0,0.0,0.0],1:[0.0,0.0,1.0,0.0],2:[0.0,0.0,0.0,1.0]}
    cdr_keys= list(cdr_ohe_dict.keys())

    if cf['Classification']=='Y':
        from scipy import stats
        learning_rate = cf['Training']['learning_rate']
        adm = optimizers.Adam(lr=learning_rate)
        cdr_prediction = prediction

        model.compile(loss='categorical_crossentropy', optimizer=adm, metrics=[tf.keras.metrics.AUC(),\
            'categorical_accuracy', 'accuracy'])
        classification_eval= model.evaluate(test_generator, steps=test_steps, verbose=1)

        for i in range(num_patients):
            cdr_pred_list=[]
            print('Predictions for patient:', total_patient_name[i])
            print('Actual value : ', test_cdr[i])
            ###########################################################################
            if samples == 20:
                
                for k in range(20):
                    pred = cdr_prediction[k+ 20*i ]
                    pred_arg= np.argmax(pred)
                    pred=cdr_keys[pred_arg]
                    cdr_pred_list.append(pred)
                    print('Prediction for slice {} is {}  |  {}'.format(k,cdr_prediction[k + (20 * i)],pred))

            ############################################################################

            if samples == 10:
                
                for k in range(8):
                    pred = cdr_prediction[k + (10 * i)]
                    pred_arg= np.argmax(pred)
                    pred=cdr_keys[pred_arg]
                    cdr_pred_list.append(pred)
                    print('Prediction for slice {} is {} |  {}'.format(k,cdr_prediction[k + (10 * i)],pred))

            ###########################################################################        

            most_freq_prediction= stats.mode(cdr_pred_list).mode

            print('Most Frequent value predicted among slices : ', most_freq_prediction)
        print(f'classification eval metrics={classification_eval}')
    else:
        age_prediction = [ item for elem in prediction for item in elem]
        for i in range(num_patients):

            print('Predictions for patient:', total_patient_name[i])
            
            if samples == 20:
                
                for k in range(20):
                    pred = age_prediction[k + (20 * i)]
                    print('Prediction for slice {} is {}'.format(k,pred))
            
            if samples == 10:
                
                for k in range(8):
                    pred = age_prediction[k + (10 * i)]
                    print('Prediction for slice {} is {}'.format(k,pred))  
        for i in range(num_patients):
     
                predicted_age_patient = []
                
                if samples == 20:
                    
                    for k in range(10): 
                        predicted_age_patient.append(age_prediction[(k+5) + (20 * i)])
                        
                    print('Details of patient {}'.format(total_patient_name[i]))
                        
                    predicted_age = numpy.mean(predicted_age_patient)
                    print('Predicted value : ', predicted_age)
                    print('Actual value : ', test_age[i])
                    
                    maximum_prediction= max(predicted_age_patient)
                    print('Max prediction : ',maximum_prediction)
                    maximum_prediction_list.append(maximum_prediction)
                    maximum_prediction_diff = maximum_prediction - test_age[i]
                    maximum_prediction_diff_list.append(maximum_prediction_diff)
                    print('The difference of max prediction from actual age: ',maximum_prediction_diff)
                    index = predicted_age_patient.index(max(predicted_age_patient )) + 5
                    print('The slice at which this maximum occured: ',index)
                    slice_max.append(index)
                    
                    minimum_prediction= min(predicted_age_patient)
                    print('Min prediction: ', minimum_prediction)
                    minimum_prediction_list.append(minimum_prediction)
                    minimum_prediction_diff = minimum_prediction - test_age[i]
                    minimum_prediction_diff_list.append(minimum_prediction_diff)
                    print('The difference of min prediction from actual age: ',minimum_prediction_diff)
                    index_min = predicted_age_patient.index(min(predicted_age_patient )) + 5
                    print('The slice at which this minimum occured: ',index_min)
                    slice_min.append(index_min)
                    print('\n')
        
                    
                    total_predicted_age_regresion.append(predicted_age)
                    
                if samples == 10:
                    
                    for k in range(8): 
                        predicted_age_patient.append(age_prediction[(k+ 1) + (10 * i)])
                        
                    print('Details of patient {}'.format(total_patient_name[i]))
                        
                    predicted_age = numpy.mean(predicted_age_patient)
                    print('Predicted value : ', predicted_age)
                    print('Actual value : ', test_age[i])
                    
                    maximum_prediction= max(predicted_age_patient)
                    print('Max prediction : ',maximum_prediction)
                    maximum_prediction_list.append(maximum_prediction)
                    maximum_prediction_diff = maximum_prediction - test_age[i]
                    maximum_prediction_diff_list.append(maximum_prediction_diff)
                    print('The difference of max prediction from actual age: ',maximum_prediction_diff)
                    index = predicted_age_patient.index(max(predicted_age_patient )) + 1
                    print('The slice at which this maximum occured: ',index)
                    slice_max.append(index)
                    
                    minimum_prediction= min(predicted_age_patient)
                    print('Min prediction: ', minimum_prediction)
                    minimum_prediction_list.append(minimum_prediction)
                    minimum_prediction_diff = minimum_prediction - test_age[i]
                    minimum_prediction_diff_list.append(minimum_prediction_diff)
                    print('The difference of min prediction from actual age: ',minimum_prediction_diff)
                    index_min = predicted_age_patient.index(min(predicted_age_patient )) + 1
                    print('The slice at which this minimum occured: ',index_min)
                    slice_min.append(index_min)
                    print('\n')
        
                    
                    total_predicted_age_regresion.append(predicted_age)

        residual = np.abs(np.array(total_predicted_age_regresion) - np.array(test_age))

        # use dataframe sort actual age and predicted age
        #A pandas data frame with columns on the left and the contents of columns on right
        data = pd.DataFrame({'patient_id': total_patient_name,
                             'actual_age': test_age,
                             'regression_predicted_age': total_predicted_age_regresion,
                             'Deviation': residual,
                             'max prediction': maximum_prediction_list,
                             'max_prediction_difference': maximum_prediction_diff_list,
                             'Slice Max': slice_max,
                             'min prediction': minimum_prediction_list,
                             'min_prediction_difference': minimum_prediction_diff_list,
                             'Slice Min': slice_min
                             })

        data1 = pd.DataFrame(data)
        #Sorts the columns by actual age
        data.sort_values(by=['actual_age'], inplace=True)
        #resets the index and drops the old index which would be added as a column to the data frame
        data.reset_index(inplace=True)
        data.drop(columns=['index'], inplace=True)
        
        data1.sort_values(by=['Deviation'], inplace=True)
        #resets the index and drops the old index which would be added as a column to the data frame
        data1.reset_index(inplace=True)
        data1.drop(columns=['index'], inplace=True)
        
        #Specify the exact path for saving csv file. Specify the correct experiment folder
        pred_path = os.path.join(os.getcwd(),'exp/'+exp_name+'/model/predictions.csv')
        print(f'Predictions will be save to{pred_path}')
        data1.to_csv(pred_path)
    
        def rmse_test(predictions, targets):
            return np.sqrt(((predictions - targets) ** 2).mean())
        
        std = np.std(np.array(data['regression_predicted_age']) - np.array(data['actual_age']))
        mae = np.mean(np.abs(np.array(data['regression_predicted_age']) - np.array(data['actual_age'])))
        rms_error = rmse_test(np.array(data['regression_predicted_age']),np.array(data['actual_age']))

        
        print('-' * 75)
        print('Test results: ')
        print('The mae for the test set is: ', mae)
        print('The std for the test set is: ', std)
        print('The rmse for the test set is: ', rms_error)
            

        # plot
        fig = plt.figure(figsize=(18, 15))
        plt.plot(data['regression_predicted_age'], marker='*', label='predicted age')
        plt.plot(data['actual_age'], marker='x', label='actual_age', )

        plt.xticks(arange(len(total_patient_name)), data['patient_id'], rotation=60)
        plt.xlabel("Patient Number")
        plt.ylabel("Age")
        plt.legend()
        plt.grid()
        plt.gcf().subplots_adjust(bottom=0.15)
        

        plt.title("Predicted age vs. Actual age, Std:{:0.2f}, MAE:{:0.2f}, RMSE:{:0.2f}".format(std, mae, rms_error))
        figurename = 'BioAge.png'
        save_file = os.path.join(cf['Paths']['model'], figurename)
        
        plt.savefig(save_file)
        print('save successful')
        plt.show()

#Preprocess the data before feeding into training network
def data_preprocess(cf):
    #Here the various paths would have already been saved into the config file. The path is now used to create directories
    #if os.path.exists return False that is if the directory has not been created then create the directory
    if not os.path.exists(cf['Paths']['save']):
        os.makedirs(cf['Paths']['save'])


    if not os.path.exists(cf['Paths']['model']):
        os.makedirs(cf['Paths']['model'])


    print('-' * 75)
    print(' Config\n')
    print(' Local saving directory : ' + cf['Paths']['save'])

    # Copy train script and configuration file (make experiment reproducible)
    shutil.copy(os.path.basename(sys.argv[0]), os.path.join(cf['Paths']['save'], 'train.py'))

    shutil.copy(cf['Paths']['config'], os.path.join(cf['Paths']['save'], 'config_Age.yml'))

    shutil.copy('./util/generator_3D_volume_slices_age_with_gender.py', os.path.join(cf['Paths']['save'], 'generator.py'))
    shutil.copy('./get_train_eval_files_multiple.py', os.path.join(cf['Paths']['save'], 'get_train_eval_files.py'))
    #shutil.copy('./network/Hybrid3DCNN_gender_age_v2.py', os.path.join(cf['Paths']['save'], 'network.py'))
    shutil.copy('./network/Hybrid3DCNN_oasis.py', os.path.join(cf['Paths']['save'], 'network.py'))
    # Extend the configuration file with new entries
    with open(os.path.join(cf['Paths']['save'], 'config_Age.yml'), "w") as ymlfile:
        yaml.dump(cf, ymlfile)



# def gradcam(model,inputs,label,base_path,slice_id):
    
#     # LAYER_NAME='conv3d_65'
#     # input_img=   tf.concat([img_chunk,gender]
#     # input_img=tf.concat(seed_input,axis=-1)
#     img_chunk = inputs[0]
#     gender = inputs[1]
    
#     # inputs[0] = tf.convert_to_tensor(inputs[0],dtype=tf.float32)
#     # inputs[1] = tf.reshape(inputs[1],[1])
#     img_chunk = tf.reshape(img_chunk,img_chunk.shape[:-1])
#     gender = tf.reshape(gender,[1])
#     print(f'input shapes received = {gender.shape,img_chunk.shape,inputs[0].shape,inputs[1].shape}')
#     grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(index=-11).output, model.output])
#     # print(f'grad_model inputs={grad_model.inputs}')

#     with tf.GradientTape() as tape:
#         conv_outputs, predictions = grad_model([img_chunk,gender])
#         loss = predictions

#     output = conv_outputs[0]
#     grads = tape.gradient(loss, conv_outputs)[0]

#     gate_f = tf.cast(output > 0, 'float32')
#     gate_r = tf.cast(grads > 0, 'float32')
#     guided_grads = tf.cast(output > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads

#     weights = tf.reduce_mean(guided_grads, axis=(0,1,2))
    
#     # print(f'weights  ={weights.shape}')

#     # cam = np.ones(output.shape[1: 4], dtype = np.float32) #image chunk shape: (1,121,145,6)

#     # for i, w in enumerate(weights):
#     #     cam += w * output[:, :, i]

#     # cam = cv2.resize(cam.numpy(), (224, 224))
    
#     cam = tf.reduce_sum(tf.multiply(output,weights),axis=-1)

#     colormap= cv2.COLORMAP_JET#16#cv2.COLORMAP_VIRIDIS
#     # print((original_image.shape[2],original_image.shape[1], original_image.shape[0]))
#     img=img_chunk.numpy()
#     heatmap = cv2.resize(np.array(cam), (img.shape[2],img.shape[1]))
#     orig_image=image_to_uint_255(img)[0,:,:,-1]
#     cam = cv2.resize(cam.numpy(), (img.shape[2], img.shape[1]))
#     cam = np.maximum(cam, 0)
#     heatmap = (cam - cam.min()) / (cam.max() - cam.min())
#     heatmaps = []

#     for i in range(heatmap.shape[-1]):
#         mri_grad_cmap_img = cv2.applyColorMap(
#             cv2.cvtColor((heatmap[:,:,i] * 255).astype("uint8"), cv2.COLOR_GRAY2BGR), colormap
#         )
#         heatmaps.append(mri_grad_cmap_img)

#     image_weight=0.7
#     mri_img =  orig_image
#     output_cams=[]

#     for i in range(heatmap.shape[-1]):

#       output = cv2.addWeighted( 
#               cv2.cvtColor(mri_img, cv2.COLOR_RGB2BGR), image_weight, heatmaps[i], 1, 0
#           )
#       output_cams.append(output)

#     for i, output in enumerate(output_cams):
#       # out_img=cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
#       # cv2_imshow(output)
#       img_path = base_path+'_chunk_'+str(slice_id)+'_gradcam_out_map'+str(i+1)+'.png'
#       print(f'img_path={img_path}')
#       cv2.imwrite(img_path,output)


   

        
    
'''
def visualize_test(cf, exp_name,vis_name='gcam'):
    test_label_path = cf['Paths']['labels']
    df = pd.read_csv(test_label_path)
    if not os.path.exists(cf['Paths']['vis']):
        os.makedirs(cf['Paths']['vis'])
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cf['Training']['gpu_num'])
    #test_tfrecord = cf['Paths']['test_tfrecord']
    batch_size = cf['Training']['batch_size']
    image_shape = cf['Training']['image_shape']

    samples = cf['Training']['samples']
    num_parallel_calls = cf['Training']['num_parallel_calls']
    case = cf['Case']

    def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 

    if cf['Pretrained_Model']['path'] is not None:
        print(' Load pretrained model')
        model = tf.keras.models.load_model(filepath=cf['Pretrained_Model']['path'], compile=False)
        print(' Summary of the model:')
        model.summary()
        
    else:
        print('no pretrained  model')
    # test_path_length = len(test_tfrecord)
    # get test Patient
    test_data_path = cf['Paths']['test_tfrecord']
    test_label_path = cf['Paths']['labels']
    test_patients,test_ids, test_age,test_gender  = get_train_eval_files.get_test_files(test_label_path,test_data_path)

    print(test_age)
    count_test = len(test_patients) * samples
    test_steps = math.ceil(count_test / batch_size)

    test_generator = generator.tfdata_generator_volume_chunks(file_lists=test_patients, label_lists=test_age,
    label_gender=test_gender,
                                                              num_parallel_calls=num_parallel_calls,
                                                                          train_patch_size = image_shape,
                                                                          samples = samples
                                                              )

    test_generator = generator.batch_and_run(test_generator, batch_size,
                                                 count_test, case=case)

    # age_prediction = model.predict(test_generator, steps=test_steps, verbose=1)
    # print(len(test_age))
    # print(len(age_prediction))
    # print(age_prediction)
    # age_prediction = [ item for elem in age_prediction for item in elem]
    
    # print(len(age_prediction))

    layers= model.layers
    print(dir(layers),dir(model))
    print(dir(layers[-1]),layers[-1].name)
    #print(model.layers.conv3d_65)
    target_layer = layers[-11]

    
    # tensor=next(test_generator)
    # with tf.GradientTape() as tape:
    #     print(f'dims={tensor[0][0][0].shape,tensor[0][1][0].shape}')
    #     conv_outputs, predictions = model([tensor[0][0][0]])
    # print(f'tensor generated={tensor}')
    # print('###################2222222######################')
    # print(f'tensor generated={tensor[0]} & {tensor[1]}')
    # print('###################3333333333######################')
    # print(f'tensor generated={tensor[0][0],tensor[0][1]} & {tensor[1][0]}')
    # print('###################44444444444######################')
    # # print(f'tensor shape={tensor}')
    # print(f'tensor generated indv={tensor[0][0][0],tensor[0][1][0]} & {tensor[1][0]}')
    # print(f'tensor generated indv={tensor[0][0][0].shape,tensor[0][1][0].shape} & {tensor[1][0].shape}')

    #**************** custom vis 
    # while True:
  
        # try:
    i=0
    subject_index =0
    total_age=0
    for tensor in test_generator:
        print(i+1)
        i+=1
        continue
        
        for j in range(len(tensor[1])):#tensor[1] is list of labels with length=batch size
            img_tensor = tf.expand_dims(tensor[0][0][j],axis=0)
            path = os.path.join(cf['Paths']['vis'],test_ids[subject_index])
            slice_id = i+1
            print(f'output base  path={path,slice_id}')
            # gradcam(model,[img_tensor,tensor[0][1][j]],tensor[1][j],path,slice_id)
            if vis_name=='sgcam_pp':
                smooth_gradcam_pp(model,[img_tensor,tensor[0][1][j]],tensor[1][j],path,slice_id)
            else:
                gradcam(model,[img_tensor,tensor[0][1][j]],tensor[1][j],path,slice_id)
            
            ## cv2.imwrite(path+'gradcam.png', output_image)
            if (i+1)%samples==0:
                # avg_age=total_age/samples
                # total_age=0
                i=0
                # print(f"True Age label for {test_ids[subject_index]} = {df[df['MR ID']==test_ids[subject_index]]['Age'].values[0]}")
                # print(f'Average Age Predicted for {test_ids[subject_index]} = {avg_age}')
                print(f'subject_index={test_ids[subject_index]} completed')
                subject_index+=1
            else:
                i+=1
                #below line makes sense only for smooth gc++ for others it will remain unaffected
                # total_age+=age

        # except tf.errors.OutOfRangeError:
        #     print('Finished')
        #     break
    return
'''
    #***************  



if __name__ == '__main__':
    #The argparse module makes it easy to write user-friendly command-line interfaces
    #first step in using the argparse is creating an ArgumentParser object
    parser = argparse.ArgumentParser(description='BioAgeNet training')

    #Adding an argument to specify the path of the configuration file
    parser.add_argument('-c', '--config_path',
                        type=str,
                        default='config/config_Age_3D.yml',
                        help='Configuration file')
    #Adding an argument to specify name of the experiment
    parser.add_argument('-e', '--exp_name',
                        type=str,
                        default=None,
                        help='Name of experiment')
    parser.add_argument('-m', '--mode',
                        type=str,
                        default='train',
                        help='Run mode train/test')
    #Adding an argument to specify vis technique
    parser.add_argument('-v', '--vis_name',
                        type=str,
                        default='gcam',
                        help='Name of Visualization technique')
    parser.add_argument('-cc', '--classify_cdr',
                        type=str,
                        default='N',
                        help='Enable cdr classification')
    #ArgumentParser parses arguments through the parse_args() method. 
    #This will inspect the command line, convert each argument to the appropriate type and then invoke the appropriate action
    arguments = parser.parse_args()

    arguments.config_path = "config/config_Age_3D.yml"

    assert arguments.config_path is not None, 'Please provide a configuration path using' \
                                              ' -c pathname in the command line.'
    assert arguments.exp_name is not None, 'Please provide a name for the experiment' \
                                           ' -e name in the command line'
    assert arguments.vis_name is not None, 'Please provide a name for the vis technique' \
                                           ' -v name in the command line'

    # Parse the configuration file
    with open(arguments.config_path, 'r') as ymlfile:
        cf = yaml.load(ymlfile)

    # Set paths (Does snot create the directory but just updates the paths in the config file)
    #Inside the Paths section, creates a new save path under the given experimentname
    cf['Paths']['save'] = os.path.join( os.getcwd(),'exp/' + arguments.exp_name)
    #Creates a directory model to save the model
    cf['Paths']['model'] = os.path.join(cf['Paths']['save'], 'model/')
    
    #Saves the path of the configuration file
    cf['Paths']['config'] = arguments.config_path

    #visualization directory:

    dt_string = datetime.now().strftime('%d-%m-%Y-%H-%M')
    cf['Paths']['vis'] = os.path.join( os.getcwd(),'exp/' + arguments.exp_name+'/'+arguments.vis_name+'/'+dt_string)

    cf['Classification'] = arguments.classify_cdr
    
    # cf['Paths']['vis'] = os.path.join( os.getcwd(),'exp/' + arguments.exp_name+'/'+arguments.vis_name+'_vis3')#gradcam_vis_results
    # create folder to store training results
    cf['Case'] = arguments.mode
    if cf['Case'] == "train":
        data_preprocess(cf)
        train(cf,arguments.exp_name)
    elif cf['Case'] == "test":
        test(cf,arguments.exp_name)
    elif cf['Case'] == "vis":
        # if arguments.vis_name == 'gcam' or arguments.vis_name == 'gcam_pp':
        visualize_test(cf,arguments.exp_name,arguments.vis_name)
        # elif arguments.vis_name == 'sgcam_pp':
        #     visualize_test(cf,arguments.exp_name)

