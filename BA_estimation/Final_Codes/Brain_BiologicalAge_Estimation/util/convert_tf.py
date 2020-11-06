#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Parse a .tfrecord file.
'''
import tensorflow as tf

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
