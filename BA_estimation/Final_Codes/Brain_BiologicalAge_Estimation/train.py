#############################################################################################################################################
# This is the main file for training the network for Biological Age estimation. The file consists of an Iterative training and testing loop.
# The input data strategy of dividing the volume into volume chunks is employed here. The hybrid 3D CNN network is used for training.
#############################################################################################################################################
##Shashank Salian : tf2 compatible version with test visualization (dev phase)
##############
import os
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
from tensorflow.compat.v1.keras.backend import set_session,clear_session,get_session
from datetime import datetime
from skimage.transform import resize

K.set_image_data_format = 'channels_last'
# from keras.backend.tensorflow_backend import set_session
# from keras.backend.tensorflow_backend import clear_session
# from keras.backend.tensorflow_backend import get_session

####
#Choose the GPU on which the training should run
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# config = tf.ConfigProto()
# #config.gpu_options.per_process_gpu_memory_fraction = 0.2
# config.gpu_options.allow_growth = True
# set_session(tf.Session(config=config))
####

print(tf.__version__)
total_gpus=tf.config.experimental.list_physical_devices('GPU')
print(f'total_gpus={total_gpus}')
gpu=total_gpus[0]
tf.config.experimental.set_visible_devices(gpu,'GPU')
#tf.config.experimental.set_memory_growth(gpu, True)

logical_gpus = tf.config.experimental.list_logical_devices('GPU')
print(f'GPUs used = {logical_gpus}')



# Import own scripts
import util.generator_3D_volume_slices_age_with_gender as generator
# import get_train_eval_files_Bio as get_train_eval_files
import get_train_eval_files_multiple as get_train_eval_files
#import evaluation_plot
#import multi_gpu
import numpy
import gc
from network import Hybrid3DCNN_gender_age_v2,Hybrid3DCNN_oasis #,Hybrid3DCNN_gender_age_v2_classification
# from network import Final_3D_volume_slice1
# from SpectralNormalizationKeras import DenseSN, ConvSN1D, ConvSN2D, ConvSN3D
# from dense import DenseNet

#Use the below class only if you want to train on several GPUs
class ModelMGPU(Model):
    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
           serial-model holds references to the weights in the multi-gpu model.
           '''
        if 'load' in attrname or 'save' in attrname:
           return getattr(self._smodel, attrname)
        else:

           return super(ModelMGPU, self).__getattribute__(attrname)


def train(cf):
    
    print(f'GPUs used = {logical_gpus}')
    
    #config = tf.compat.v1.ConfigProto()
    #config.gpu_options.allow_growth = True

    #train_path = cf['Paths']['train_tfrecord']
    train_eval_ratio = cf['Data']['train_val_split']
    batch_size = cf['Training']['batch_size']
    image_shape = cf['Training']['image_shape']
    num_parallel_calls = cf['Training']['num_parallel_calls']
    epoch = cf['Training']['num_epochs']
    unhealthy_path = cf['Paths']['unhealthy']+ '/'+ cf['Paths']['exp']
    train_data_path = cf['Paths']['train_tfrecord']
    train_label_path = cf['Paths']['labels']
    samples = cf['Training']['samples']
    #This variable is initially set to True to start the iteration process.
    continueTraining = True
   
    #Count the number of iterations
    counter = 0
   
    while(continueTraining):
        
        counter = counter + 1
    	#################### new code ######################
        K.clear_session()

    	##########################################
        #Divide the entire dataset into training and validation set
        # train_patient, train_labels, train_labels_gender, eva_files, eva_labels, eva_labels_gender = get_train_eval_files.prepare_train_eval_files(train_path, train_eval_ratio)
        train_patients,train_labels, train_labels_gender, train_cdr, train_scan_ids,eva_patients,eva_labels, eva_labels_gender,eval_cdr,eval_scan_ids \
         = get_train_eval_files.prepare_train_eval_files(train_label_path,train_data_path, train_eval_ratio)

        total_patient_name1 = []
        #Get the validation patients names
        for i in range(len(eva_patients)):
            # validation_pateint_number1= eva_patients[i][46:-9]
            validation_pateint_number1= eval_scan_ids[i]
            total_patient_name1.append(validation_pateint_number1)
       
        age_labels = []
        for i in range(len(eva_patients)):
           validation_pateint_age_labels= eva_labels[i]
           age_labels.append(validation_pateint_age_labels)
            
        gender_labels = []
        for i in range(len(eva_patients)):
           validation_pateint_gender_labels= eva_labels_gender[i]
           gender_labels.append(validation_pateint_gender_labels)
    
        total_patient_name2 = []
        #Get the training patients names
        for i in range(len(train_patients)):
            # validation_pateint_number2= train_patients[i][46:-9]
            validation_pateint_number2=train_scan_ids[i]
            total_patient_name2.append(validation_pateint_number2)
        
        print('Length of training patients is: ', len(train_patients))
        print('The training patients are: ',total_patient_name2)
        print('Length of validation patients is: ', len(eva_patients)) 
        print('The validation patients are: ',total_patient_name1)
        print('The ages of validation patients are: ',age_labels)
        print('The genders of validation patients are: ',gender_labels)

        #The total count has to be multiplied by the number of samples per patient. In this case it is 20 volume chunks per patient
        count_train= len(train_patients) * 20
        count_validation = len(eva_patients) * 20

        steps_per_epoch = math.ceil(count_train / batch_size) + 1
        validata_steps = math.ceil(count_validation / batch_size) + 1

        print('Expected training steps: ', steps_per_epoch)
        print('Expected validation_steps: ', validata_steps)
        print('Batch size : ', batch_size)

        print('-' * 75)
        print(' Model\n')

        #Create the model with the shape of the input
        input_size = image_shape + [1]
        #model,_ = Hybrid3DCNN_gender_age_v2.createModel(input_size)
        model,_ = Hybrid3DCNN_oasis.createModel(input_size)
        #Uncomment the below command in case of using multiple GPUs
        #model = ModelMGPU(model, 2)
        
        #Defining RMSE metric
        def rmse(y_true, y_pred):
            return K.sqrt(K.mean(K.square(y_pred - y_true))) 
    
        learning_rate = cf['Training']['learning_rate']
        
        #Uncomment below commands if using SGD optimizer
        #sgd = optimizers.SGD(lr=0.0001,clipnorm=25.0, decay=1e-6, momentum=0.9, nesterov=True)
        #model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['mae'])
        
        print('learning rate is:',learning_rate )
        adm = optimizers.Adam(lr=learning_rate)
        model.compile(loss='mse', optimizer=adm, metrics=['mae', rmse])
        model.summary()
        print(' Model compiled!')
   
        #Defining the callback functions
        def get_callbacks(model_file, logging_file):
            callbacks = list()
            #Save the model
            callbacks.append(cb.ModelCheckpoint(model_file, monitor='val_mae', save_best_only=True, mode='min'))
            #Save the log file
            callbacks.append(CSVLogger(logging_file, append=True))
            #Reduce LR on plateau
            callbacks.append(ReduceLROnPlateau(monitor='val_mae', factor=0.2, patience=12, min_lr=1e-7))
            #Stop training in case of validation error increase
            callbacks.append(EarlyStopping(monitor='val_mae', min_delta=0.005, patience=18, verbose=1, mode='auto', baseline=None, restore_best_weights=False))

            return callbacks

        print('-' * 75)
        print(' Training...')

        #Use a custom data generator to generate volume chunks from a complete volume of a patient
        
        #For the training patients
        train_generator = generator.tfdata_generator_volume_chunks(file_lists=train_patients,
                                                                          label_lists=train_labels,
                                                                          label_gender=train_labels_gender,
                                                                          num_parallel_calls=num_parallel_calls,\
                                                                          train_patch_size = image_shape,\
                                                                          samples = samples

                                                                          )

        train_generator = generator.batch_and_run(train_generator, batch_size, count_train, case='train')

        #For the validation patients
        val_generator = generator.tfdata_generator_volume_chunks(file_lists=eva_patients,
                                                                        label_lists=eva_labels,
                                                                        label_gender=eva_labels_gender,
                                                                        num_parallel_calls=num_parallel_calls,\
                                                                        train_patch_size = image_shape,
                                                                        samples = samples
                                                                        )


        val_generator = generator.batch_and_run(val_generator, batch_size, count_validation, case='valid')

        path_w = os.path.join(cf['Paths']['model'] , "age_net_oasis1_3.hdf5")
        logging_file = os.path.join(cf['Paths']['model'] , "age_net_oasis1_3.txt")

        #Start training the model
        res = model.fit(
              train_generator,
              steps_per_epoch=steps_per_epoch,
              epochs = epoch,
              validation_data=val_generator,
              validation_steps=validata_steps,
              callbacks=get_callbacks(model_file=path_w, logging_file=logging_file))
        print(f'training history={res.history}')
        #After training the model, reset the Keras session in order to do inference on the validation set
        #used k.clear_session() at loop start instead of old reset_keras()
        #### reset_keras()
        
        batch_size = 10
        
        count_validation = len(eva_patients) * 20
        
        validata_steps = math.ceil(count_validation / batch_size)
        #val_generator = generator.tfdata_generator_volume_chunks(file_lists=eva_patients,
         #                                                               label_lists=eva_labels,
          #                                                              label_gender=eva_labels_gender,
           #                                                             num_parallel_calls=num_parallel_calls,

            #                                                            )



        #val_generator = generator.batch_and_run(val_generator, batch_size, count_validation, case='valid')

        model = tf.keras.models.load_model(filepath=cf['Pretrained_Model']['path'], compile=False)
        
        #Predict on the validation set
        age_prediction = model.predict_generator(val_generator, steps=validata_steps, verbose=1)
        age_prediction = [ item for elem in age_prediction for item in elem]
        
        print('Length of validation patient is:', len(eva_labels))
        print('Length of predictions is :', len(age_prediction))
        print('The predictions are: ',age_prediction)
        
        total_patient_name = []
        for i in range(len(eva_patients)):
            # validation_pateint_number = eva_patients[i][46:-9]
            validation_pateint_number = eval_scan_ids[i]
            total_patient_name.append(validation_pateint_number)

        total_predicted_age_regresion = []
        maximum_prediction_list = []
        minimum_prediction_list = []
        minimum_prediction_diff_list = []
        maximum_prediction_diff_list = []
        slice_max = []
        slice_min = []
        max_deviation_list = []
        
        #Calculate the predictions for each patient as an average of 10 volume chunks
        for i in range(len(eva_labels)):

            predicted_age_patient = []
            deviation_list = []
            
            for k in range(10):
   
                deviation = abs(age_prediction[(k+5) + (20 * i)] - eva_labels[i])
                deviation_list.append(deviation)  
                predicted_age_patient.append(age_prediction[(k+5) + (20 * i)])
            
            print('Details of patient {}'.format(total_patient_name[i]))
            print('The deviations of patient are :', deviation_list)
            
            #Maximum deviation for a patient out of the 10 chunks
            maximum_deviation = max(deviation_list)
            print('Max deviation:', maximum_deviation)
            max_deviation_list.append(maximum_deviation)
            
            #Mean of predictions of 10 volume chunks = prediction of the patient
            predicted_age = numpy.mean(predicted_age_patient)
            print('The predicted values are: ', predicted_age_patient)
            print('Predicted value : ',predicted_age)
            print('Actual value :  ', eva_labels[i])
                    
            #Maximum of the volume chunk predictions 
            maximum_prediction= max(predicted_age_patient)
            print('Max prediction : ',maximum_prediction)
            maximum_prediction_list.append(maximum_prediction)
            maximum_prediction_diff = maximum_prediction - eva_labels[i]
            maximum_prediction_diff_list.append(maximum_prediction_diff)
            print('The difference of max prediction from actual age: ',maximum_prediction_diff)
            index = predicted_age_patient.index(max(predicted_age_patient )) + 5
            print('The slice at which this maximum occured: ',index)
            slice_max.append(index)
            
            #Minimum of the volume chunk predictions 
            minimum_prediction= min(predicted_age_patient)
            print('Min prediction: ', minimum_prediction)
            minimum_prediction_list.append(minimum_prediction)
            minimum_prediction_diff = minimum_prediction - eva_labels[i]
            minimum_prediction_diff_list.append(minimum_prediction_diff)
            print('The difference of min prediction from actual age: ',minimum_prediction_diff)
            index_min = predicted_age_patient.index(min(predicted_age_patient )) + 5
            print('The slice at which this minimum occured: ',index_min)
            slice_min.append(index_min)
            print('\n')
            
            total_predicted_age_regresion.append(predicted_age)
            
        print('The validation patients are:')
        print(total_patient_name)
        print('\n')
        print('The actual ages are: ')
        print(eva_labels)
        print('\n')
        print('The predicted ages are: ')
        print(total_predicted_age_regresion)
        print('\n')
        print('The maximum deviations of the patient are: ')
        print(max_deviation_list)
        print('\n')
        
        residual = np.abs(np.array(total_predicted_age_regresion) - np.array(eva_labels))
        
        data = pd.DataFrame({'patient_id': total_patient_name,
                         'actual_age': eva_labels,
                         'regression_predicted_age': total_predicted_age_regresion,
                         'Deviation': residual,
                         'max prediction': maximum_prediction_list,
                         'max_prediction_difference': maximum_prediction_diff_list,
                         'Slice Max': slice_max,
                         'min prediction': minimum_prediction_list,
                         'min_prediction_difference': minimum_prediction_diff_list,
                         'Slice Min': slice_min
                         })

    
        #Sorts the columns by their deviations
        data.sort_values(by=['Deviation'], inplace=True)
        #resets the index and drops the old index which would be added as a column to the data frame
        data.reset_index(inplace=True)
        data.drop(columns=['index'], inplace=True)
        
        
        def rmse_test(predictions, targets):
            return np.sqrt(((predictions - targets) ** 2).mean())
    
        print(data)
        data1 = pd.DataFrame(data)
        #Saves the predictions for every iteration
        filename = 'predictions' + str(counter) + '.csv'
        print(filename)
        filename = os.path.join(cf['Paths']['model'], filename)
        data1.to_csv(filename)
        
        #Calculate the metrics on the predictions
        std = np.std(np.array(data['regression_predicted_age']) - np.array(data['actual_age']))
        mae = np.mean(np.abs(np.array(data['regression_predicted_age']) - np.array(data['actual_age'])))
        rms_error = rmse_test(np.array(data['regression_predicted_age']),np.array(data['actual_age']))
         
      
        print('-' * 75)
        print('Test results: ')
        print('\n')
        print('The mae for the test set is: ', mae)
        print('\n')
        print('The std for the test set is: ', std)
        print('\n')
        print('The rmse for the test set is: ', rms_error)
        print('\n')
        
        
        #Calculation of Thresholds
        threshold_1 = max(max_deviation_list)
        threshold_2 = numpy.mean(max_deviation_list)
        print('The threshold1 value is:', threshold_1)
        print('The threshold2 value is:', threshold_2)
        print('\n')
        
        threshold_max = (threshold_1 + mae)/2
        threshold_mean = (threshold_2 + mae)/2
        print('The threshold_max value is:', threshold_max)
        print('The threshold_mean value is:', threshold_mean)
        print('\n')
        
        #Stopping condition : If MAE < 1
        if(mae <= 1):

            continueTraining = False
            print('Iterative Training Finished!')
        
        else:
            
            #Outlier Detection
            unhealthy_list = []
        
            #For every patient in the validation set, check if it is a outlier
            for i in range(len(eva_labels)):
                
                diff_list = []
                setFlag = False
                
                print('Checking patient:',total_patient_name[i])
                
                for k in range(10):
                         
                    diff = abs(age_prediction[(k+5) + (20 * i)] - eva_labels[i])
                    diff_list.append(diff)
                    
                    #Check deviation of each volume chunk of a patient
                    #If it exceeds the max threshold, Set the flag to true
                    if(diff >= threshold_max):
                        print('Diff greater than threshold')
                        print('The value which exceeded threshold is', diff)
                        print('The slice which exceeded threshold is', (k+5))
                        setFlag = True
                                
                print('Deviation list is', diff_list)

                mean_diff = numpy.mean(diff_list)
                print('The mean of deviations :', mean_diff)

                #If the mean of deviations of the volume chunks exceeds the threshold and the flag is set, the mark patient as outlier
                if((mean_diff >= threshold_mean) & setFlag):
                        print('Mean deviation of patient exceeded threshold as well as max deviation per slice exceeded')
                        unhealthy_Patient = total_patient_name[i]
                        print('The unhealthy patient is', unhealthy_Patient)
                        print('\n')
                        unhealthy_list.append(unhealthy_Patient)
                
                # If only the mean threshold exceeded, patient is considered normal        
                elif(mean_diff >= threshold_mean):
                        print('Only the mean deviation exceeded threshold')
                        print('Patient is considered healthy')
                       
                # If only the max threshold exceeded, patient is considered normal        
                elif(setFlag):
                        print('Only max deviation of slice exceeded')
                        print('Patient is healthy')
                        print('\n')
                      

            # List of unhealthy patients
            print('Length of unhealthy patients :', len(unhealthy_list))
            print('\n')
            print('Unhealthy patients are: ')
            print('\n')
            print(unhealthy_list)
        
            #If the unhealthy list > 0, then remove the patient from the dataset    
            if not os.path.exists(unhealthy_path):
            	os.makedirs(unhealthy_path)
            if len(unhealthy_list) > 0:
                
                for unhealthy_patient in unhealthy_list:
                   
                    unhealthy_patient = unhealthy_patient + '.tfrecord'
                    print('moving: ',unhealthy_patient)
                    shutil.move(os.path.join(train_path, unhealthy_patient), unhealthy_path)
                
                continueTraining = True
                print('Reiterating...', counter+1)
            
            else:
            #If no patients are unhealthy, then stop iterative process    
                continueTraining = False
                
        


#Testing on the test set
def test(cf):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cf['Training']['gpu_num'])
    test_tfrecord = cf['Paths']['test_tfrecord']
    batch_size = cf['Training']['batch_size']
    num_parallel_calls = cf['Training']['num_parallel_calls']
    case = cf['Case']

    def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 

    if cf['Pretrained_Model']['path'] is not None:
        print(' Load pretrained model')
        model = keras.models.load_model(filepath=cf['Pretrained_Model']['path'], compile=False)
        print(' Summary of the model:')
        model.summary()
        
    else:
        
        print('no pretrained  model')

    # get test Patient
    # test_patient, test_age  = get_train_eval_files.get_test_files(test_path=test_tfrecord)

    test_patients,test_ids, test_labels,test_gender,test_cdr  = get_train_eval_files.get_test_files(test_path=test_tfrecord)

    print(test_age)
    count_test = len(test_patient) * 20
    test_steps = math.ceil(count_test / batch_size)

    test_generator = generator.tfdata_generator_volume_chunks(file_lists=test_patient, label_lists=test_age,
                                                              num_parallel_calls=num_parallel_calls,

                                                              )

    test_generator = generator.batch_and_run(test_generator, batch_size,
                                                 count_test, case=case)

    age_prediction = model.predict_generator(test_generator, steps=test_steps, verbose=1)
    print(len(test_age))
    print(len(age_prediction))
    print(age_prediction)
    age_prediction = [ item for elem in age_prediction for item in elem]


    print(f'All predictions={age_prediction}')
    


    total_patient_name = [ id.split('.')[0] for id in test_ids]
    num_patients = len(test_patients)
    # for i in range(len(test_patient)):
    #     test_pateint_number = test_patient[i][test_path_length:-9]
    #     total_patient_name.append(test_pateint_number)

    print(f'total_patient_name={total_patient_name}')
    print(f'test_ids={test_ids}')
    print(f'num_patients={num_patients}')


    # total_patient_name = []
    
    # for i in range(len(test_patient)):
    #     test_pateint_number = test_patient[i][44:-9]
    #     total_patient_name.append(test_pateint_number)

    for i in range(len(test_patient)):

            print('Predictions for patient:', total_patient_name[i])
            
            for k in range(20):
                pred = age_prediction[k+ (20 *i)]
                print('Prediction for slice {} is {}'.format(k,pred))
               

    # convert predicted age to float
    total_predicted_age_regresion = []
    maximum_prediction_list = []
    minimum_prediction_list = []
    minimum_prediction_diff_list = []
    maximum_prediction_diff_list = []
    slice_max = []
    slice_min = []
    
    for i in range(len(test_patient)):
 
            predicted_age_patient = []
            
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
    
    
    data1.to_csv(cf['Paths']['model'] +'/unhealthy_data_valid.csv')
    
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
    else:
        if not cf['Training']['background_process']:
            stop = input('\033[93m The folder {} already exists. Do you want to overwrite it ? ([y]/n) \033[0m'.format(
                cf['Paths']['save']))
            if stop == 'n':
                return

    if not os.path.exists(cf['Paths']['model']):
        os.makedirs(cf['Paths']['model'])
    else:
        if not cf['Training']['background_process']:
            stop = input('\033[93m The folder {} already exists. Do you want to overwrite it ? ([y]/n) \033[0m'.format(
                cf['Paths']['model']))
            if stop == 'n':
                return

#    if not os.path.exists(cf['Paths']['histories']):
#        os.makedirs(cf['Paths']['histories'])
#    else:
#        if not cf['Training']['background_process']:
#            stop = input('\033[93m The folder {} already exists. Do you want to overwrite it ? ([y]/n) \033[0m'.format(
#                cf['Paths']['histories']))
#            if stop == 'n':
#                return

    print('-' * 75)
    print(' Config\n')
    print(' Local saving directory : ' + cf['Paths']['save'])

    # Copy train script and configuration file (make experiment reproducible)
    shutil.copy(base_path+os.path.basename(sys.argv[0]), os.path.join(cf['Paths']['save'], 'train.py'))

    shutil.copy(cf['Paths']['config'], os.path.join(cf['Paths']['save'], 'config_Age.yml'))

    shutil.copy(base_path+'util/generator_3D_volume_slices_age_with_gender.py', os.path.join(cf['Paths']['save'], 'generator.py'))
    shutil.copy(base_path+'get_train_eval_files_multiple.py', os.path.join(cf['Paths']['save'], 'get_train_eval_files.py'))
    #shutil.copy(base_path+'network/Hybrid3DCNN_gender_age_v2.py', os.path.join(cf['Paths']['save'], 'network.py'))
    shutil.copy(base_path+'./network/Hybrid3DCNN_oasis.py', os.path.join(cf['Paths']['save'], 'network.py'))
    # Extend the configuration file with new entries
    with open(os.path.join(cf['Paths']['save'], 'config_Age.yml'), "w") as ymlfile:
        yaml.dump(cf, ymlfile)


if __name__ == '__main__':
    #The argparse module makes it easy to write user-friendly command-line interfaces
    #first step in using the argparse is creating an ArgumentParser object
    parser = argparse.ArgumentParser(description='BioAgeNet training')
    base_path='/usrhomes/g009/shashanks/Master_Thesis_BA_DeepVis/BA_estimation/Final_Codes/Brain_BiologicalAge_Estimation/'
    sys.path.append('/usrhomes/g009/shashanks/Master_Thesis_BA_DeepVis/BA_estimation/Final_Codes/Brain_BiologicalAge_Estimation/')
    #Adding an argument to specify the path of the configuration file
    parser.add_argument('-c', '--config_path',
                        type=str,
                        default=base_path+'config/config_Age_3D.yml',
                        help='Configuration file')
    #Adding an argument to specify name of the experiment
    parser.add_argument('-e', '--exp_name',
                        type=str,
                        default=None,
                        help='Name of experiment')
    #ArgumentParser parses arguments through the parse_args() method. 
    #This will inspect the command line, convert each argument to the appropriate type and then invoke the appropriate action
    arguments = parser.parse_args()

    arguments.config_path = base_path+"config/config_Age_3D.yml"

    assert arguments.config_path is not None, 'Please provide a configuration path using' \
                                              ' -c pathname in the command line.'
    assert arguments.exp_name is not None, 'Please provide a name for the experiment' \
                                           ' -e name in the command line'

    # Parse the configuration file
    with open(arguments.config_path, 'r') as ymlfile:
        cf = yaml.load(ymlfile)

    # Set paths (Does snot create the directory but just updates the paths in the config file)
    #Inside the Paths section, creates a new save path under the given experimentname
    cf['Paths']['save'] = base_path+'exp/' + arguments.exp_name
    #Creates a directory model to save the model
    cf['Paths']['model'] = os.path.join(cf['Paths']['save'], 'model/')
    
    #Saves the path of the configuration file
    cf['Paths']['config'] = arguments.config_path

    cf['Paths']['exp'] = arguments.exp_name

    # create folder to store training results

    if cf['Case'] == "train":
        data_preprocess(cf)
        train(cf)
    else:
        test(cf)
