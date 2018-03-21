import tensorflow as tf
import os
from PIL import Image 
import numpy as np


class Donkey(object):
    @staticmethod
    def _preprocess(image):
#        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
#        image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
#        image = tf.multiply(tf.subtract(image, 0.5), 2)
#        image = tf.reshape(image, [64, 64, 3])
#        image = tf.random_crop(image, [54, 54, 3])
#        image = tf.image.random_brightness(image,max_delta=20./255)
#        image = tf.image.random_contrast(image,0.1,0.2)
        image = tf.cast(image,tf.float32)*(1.0/255)-0.5
        image = tf.reshape(image, [28, 28, 1])
#        image = tf.image.rgb_to_grayscale
        return image

    @staticmethod
    def _read_and_decode(filename_queue):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'digit': tf.FixedLenFeature([], tf.int64),
            })

        image = Donkey._preprocess(tf.decode_raw(features['image'], tf.uint8))
        
        digit = tf.cast(features['digit'], tf.int32)
        
        return image,digit

    
    @staticmethod
    def build_batch(path_to_tfrecords_file,num_examples, batch_size, shuffled,num_epoch_):
        assert tf.gfile.Exists(path_to_tfrecords_file), '%s not found' % path_to_tfrecords_file

        filename_queue = tf.train.string_input_producer([path_to_tfrecords_file], shuffle=True,num_epochs=num_epoch_)
        image,digit = Donkey._read_and_decode(filename_queue)

        min_queue_examples = int(0.4 * num_examples)
        if shuffled:
            image_batch, digit_batch = tf.train.shuffle_batch([image,digit],
                                                               batch_size=batch_size,
                                                               num_threads=2,
                                                               capacity=min_queue_examples + 3 * batch_size,
                                                               min_after_dequeue=min_queue_examples)
        else:
            image_batch,digit_batch = tf.train.batch([image,digit],
                                                        batch_size=batch_size,
                                                        num_threads=2,
                                                        capacity=min_queue_examples + 3 * batch_size)
        return image_batch,digit_batch
    
    
    
#    
#    @staticmethod
#    def code_and_write(path_to_val_file,ftrecordfiledir):
#          cwd = path_to_val_file
#          num = 0
#          classes=['0','1','2','3','4','5','6','7','8','9','10']
#          writer= tf.python_io.TFRecordWriter(ftrecordfiledir)
#          for index,name in enumerate(classes):
##             print(index)
##             print(name)
#             class_path=cwd+name+'/'
#             for img_name in os.listdir(class_path): 
#                 num=num+1
#                 img_path = class_path+img_name
#                 img=Image.open(img_path,'r')
#                 img = img.resize((28,28))
##                 img_raw=img.tobytes()
#                 img_raw = np.array(img).tobytes()
#                 example = tf.train.Example(
#                      features=tf.train.Features(feature={
#                    'digit': tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
#                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
#                 })) 
#                 writer.write(example.SerializeToString())  
#          writer.close()
#          return num
#    
#    
#    @staticmethod
#    def build_batch_from_file(path_to_val_file,batch_size,shuffled,num_epoch_):
#        ftrecordfilename = "val_1.tfrecords"
#        ftrecordfiledir = os.path.join(path_to_val_file, ftrecordfilename)
#        num_examples = Donkey.code_and_write(path_to_val_file,ftrecordfiledir) 
#        filename_queue = tf.train.string_input_producer([ftrecordfilename], shuffle=True,num_epochs=num_epoch_)
#        image,digit = Donkey._read_and_decode(filename_queue)
#
#        min_queue_examples = int(0.4 * num_examples)
#        if shuffled:
#            image_batch, digit_batch = tf.train.shuffle_batch([image,digit],
#                                                               batch_size=batch_size,
#                                                               num_threads=2,
#                                                               capacity=min_queue_examples + 3 * batch_size,
#                                                               min_after_dequeue=min_queue_examples)
#        else:
#            image_batch,digit_batch = tf.train.batch([image,digit],
#                                                        batch_size=batch_size,
#                                                        num_threads=2,
#                                                        capacity=min_queue_examples + 3 * batch_size)
#        return image_batch,digit_batch
#        
#        
#    
    

      
