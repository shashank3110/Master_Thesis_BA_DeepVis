import logging
logging.getLogger('tensorflow').disabled = True
import tensorflow as tf
#import  torch
print('hello world')
print(f'tf version ={tf.__version__}')
total_gpus=tf.config.experimental.list_physical_devices('GPU')
print(f'total_gpus={total_gpus}')
gpu=total_gpus[1]
tf.config.experimental.set_visible_devices(gpu,'GPU')
tf.config.experimental.set_memory_growth(gpu, True)

logical_gpus = tf.config.experimental.list_logical_devices('GPU')
print(f'GPUs used = {logical_gpus}')
