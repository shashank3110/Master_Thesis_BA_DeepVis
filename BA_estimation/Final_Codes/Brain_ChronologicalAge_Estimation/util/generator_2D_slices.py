import tensorflow as tf
from util import convert_tf


def get_slice_group_2dimenstion(image=None, train_patch_size=[121, 145, 12], num_slices_per_group=10):
    offset = 0

    slice_group = []

    for i in range(num_slices_per_group):
        print(offset)
        slice_image = tf.slice(image, [0, 0, offset], train_patch_size)

        slice_group.append(slice_image)

        # print(vol_slice)
        # vol_slice = tf.expand_dims(vol_slice, -1)
        # print(vol_slice)
        # slice_group.append(vol_slice)
        # print(slice_group)
        offset = offset + 12

    # slice_concat = tf.concat(slice_group, axis=-1)
    # slice_collection.append(slice_concat)
    slice_collection = tf.stack(slice_group)
    print(slice_collection)
    assert slice_collection.get_shape().dims == [num_slices_per_group] + train_patch_size


    return slice_collection

def tfdata_generator_multi_2DSlice(file_lists, label_lists, num_parallel_calls, slice_shape, num_slices_per_group):


    #file_lists contain list of tf_record files ie the MRI volumes (converted from dicom to tfrecord)
    #This now Creates a TFRecordDataset to read one or more TFRecord files
    
     # TFRecordDataset opens a binary file and reads one record at a time.
    # `file_lists` could also be a list of filenames, which will be read in order
    filenames = tf.data.TFRecordDataset(file_lists)
    #Now the tf record files are stored in the form of as tf.Dataset
    
    #Creates a Dataset whose elements are slices of the given tensors.
    labels = tf.data.Dataset.from_tensor_slices(label_lists)

    #Creates a Dataset by zipping together the given datasets.
    #That is merges a datset -Filenames with other datasets - gender,weight...
    #dataset structure will be {[filename1,gender1,weight1,height1,labels1], [filename2,gender2,weight2,height2,labels2],.....[filename400,gender400,weight400,height400,labels400]}
    dataset = tf.data.Dataset.zip((filenames, labels))
    
    #This transformation applies map_func to each element of this dataset, and returns a new dataset containing the transformed elements, in the same order as they appeared in the input.
    #This now applies a transformation function to convert the tfrecord dataset MRi volumes into actual MRI volumes
    #num_parallel_calls is the number of files that can read in parallel. The files are interleaved
    dataset = dataset.map(
        map_func=lambda a, b: (convert_tf.parse_function_image(a), b),
        num_parallel_calls=num_parallel_calls)

    get_slice_fn = lambda image: get_slice_group_2dimenstion(image,
                                                             train_patch_size=slice_shape,
                                                             num_slices_per_group=num_slices_per_group
                                                             )
    #Now using the map function of dataset, the function of extracting 2D slices is applied to every MRI volume in the dataset
    dataset = dataset.map(map_func=lambda a, b: (get_slice_fn(a),
                                                [b for _ in range(num_slices_per_group)]
                                                ), num_parallel_calls=num_parallel_calls)
    dataset = dataset.apply(tf.contrib.data.unbatch())

    return dataset

def merge_all_dataset(dataset_1, batch_size, count, case):

    if case == 'train':

        dataset_choosed = dataset_1.shuffle(buffer_size=count)
    else:
        dataset_choosed = dataset_1

    # Combines consecutive elements of this dataset into batches.
    #Returns a dataset converted into batches
    dataset_choosed = dataset_choosed.batch(batch_size=batch_size)

    dataset_choosed = dataset_choosed.repeat()
    dataset_choosed = dataset_choosed.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

    #Once you have built a Dataset to represent your input data, the next step is to create an Iterator to access elements from that dataset
    iterator_all = dataset_choosed.make_one_shot_iterator()
    next_all = iterator_all.get_next()

    with tf.Session() as sess:
        while True:
            try:
                features, labels = sess.run(next_all)
                yield features, labels
                #print(features.shape)

            except tf.errors.OutOfRangeError:
                print('Finished')
                break
