#############################################################################################################################################
#This file is a custom data generator. It takes the set of training / validation / testing patients and returns a tf record dataset.
# Each volume of a patient is divided into volume chunks. The total number of chunks and the dimension of each chunk can be specified.
#############################################################################################################################################

import tensorflow as tf
from util import convert_tf
import numpy as np
# from tensorflow import feature_column

#Function to divide a volume into volume chunks
def get_slice_group_3dimenstion(image=None, train_patch_size=[121, 145, 6], samples = 20):
    
    offset = 0
    
    # samples = 20
    slice_group = []
   
    if samples == 10:
        add_offset = 12
    elif samples == 20:
        add_offset = 6
    else:
        add_offset = 0
 
        
    for i in range(samples):
        
        vol_slice = tf.slice(image, [0, 0, offset], train_patch_size)
        
        slice_group.append(vol_slice)
        
        offset = offset + add_offset
        
    slice_collection = tf.stack(slice_group)

    assert slice_collection.get_shape().dims == [samples] + train_patch_size
    print(slice_collection)

    return slice_collection

#Accepts a list of files of patients data and returns a tfrecord dataset
def tfdata_generator_volume_chunks(file_lists, label_lists,label_gender, num_parallel_calls,train_patch_size, samples):

    #Function for input normalization ( Zero mean and Unit variance)
    def normalize_with_moments(x, axes=[0, 1, 2], epsilon=1e-8):
        mean, variance = tf.nn.moments(x, axes=axes)
        x_normed = (x - mean) / tf.sqrt(variance + epsilon)
        return x_normed
    print(len(file_lists),len(label_lists),len(label_gender))
    
    #file_lists contain list of tf_record files ie the MRI volumes (converted from dicom to tfrecord)
    #This now Creates a TFRecordDataset to read one or more TFRecord files
    
     # TFRecordDataset opens a binary file and reads one record at a time.
    # `file_lists` could also be a list of filenames, which will be read in order
    filenames = tf.data.TFRecordDataset(file_lists)
    #Now the tf record files are stored in the form of as tf.Dataset
    
    #Creates a Dataset whose elements are slices of the given tensors. 
    
    labels = tf.data.Dataset.from_tensor_slices(label_lists)
    
    gender = tf.data.Dataset.from_tensor_slices(label_gender)

    print(f'gender={gender}')
    print(f'labels={labels}')
    

    #Creates a Dataset by zipping together the given datasets.
    #That is merges a datset -Filenames with other datasets - gender,weight...
 
    dataset = tf.data.Dataset.zip((filenames, gender,labels))
    print(dataset)
    
    #This transformation applies map_func to each element of this dataset, and returns a new dataset containing the transformed elements, in the same order as they appeared in the input.
    #This now applies a transformation function to convert the tfrecord dataset MRi volumes into actual MRI volumes
    
    dataset = dataset.map(
        map_func=lambda a, b, c: (convert_tf.parse_function_image(a), b, (tf.cast(c, tf.float32))),
        num_parallel_calls=num_parallel_calls)


    get_slice_fn = lambda image: get_slice_group_3dimenstion(image, train_patch_size=train_patch_size, samples = samples)

    dataset = dataset.map(map_func=lambda a, b, c: (get_slice_fn(a),
                                                [b for _ in range(samples)],
                                                [c for _ in range(samples)],
                                                
                                                ), num_parallel_calls=num_parallel_calls) #[d for _ in range(samples)],
    dataset = dataset.apply(tf.data.experimental.unbatch())
    
    
    return dataset


def batch_and_run(dataset_1, batch_size, count, case):

    if case == 'train':

        dataset_choosed = dataset_1.shuffle(buffer_size=count)
   
    elif case == 'valid':
    
        dataset_choosed = dataset_1.shuffle(buffer_size=count)
   
    else:
    
        dataset_choosed = dataset_1

    # Combines consecutive elements of this dataset into batches.
    #Returns a dataset converted into batches
    dataset_choosed = dataset_choosed.batch(batch_size=batch_size)
   
    dataset_choosed = dataset_choosed.repeat()
    dataset_choosed = dataset_choosed.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    #Once you have built a Dataset to represent your input data, the next step is to create an Iterator to access elements from that dataset
    iterator_all = dataset_choosed.__iter__()
    next_all = iterator_all.get_next()

    #####Shashank Salian : removed tf1 session 
    
    while True:
        try:
            
            
            features, genders, labels = next_all
            next_all = iterator_all.get_next()

            features = np.expand_dims(features, -1)

            yield [features, genders], labels 

        except tf.errors.OutOfRangeError:
            print('Finished')
            break
