import os 
import tensorflow as tf 
from PIL import Image 
import numpy as np 
from meta import Meta



recordfilenum = 0

classes=['0','1','2','3','4','5','6','7','8','9','10']

train_ftrecordfilename = ("train.tfrecords")
val_ftrecordfilename = ("val.tfrecords")
meta_filename = ('meta.json')


def create_tfrecords_meta_file(num_train_examples,num_val_examples, num_test_examples,path_to_tfrecords_meta_file):
    print 'Saving meta file to %s...' % path_to_tfrecords_meta_file
    meta = Meta()
    meta.num_train_examples = num_train_examples
    meta.num_val_examples = num_val_examples
    meta.num_test_examples = num_test_examples
    meta.save(path_to_tfrecords_meta_file)

def main(_):
    print "convert training data"
    train_cwd = './data/train_data/'
    train_num = 0
    path_to_train_tfrecords_file = train_cwd+train_ftrecordfilename
    assert not os.path.exists(path_to_train_tfrecords_file), 'The file %s already exists' % path_to_train_tfrecords_file
    writer= tf.python_io.TFRecordWriter(path_to_train_tfrecords_file)
     
    for index,name in enumerate(classes):
#        print(index)
#        print(name)
        class_path=train_cwd+name+'/'
        for img_name in os.listdir(class_path): 
            train_num=train_num+1
            img_path = class_path+img_name 
            img=Image.open(img_path,'r')
            img = img.resize((28,28))
            img_raw = np.array(img).tobytes()
            example = tf.train.Example(
                 features=tf.train.Features(feature={
                'digit': tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            })) 
            writer.write(example.SerializeToString())
            print '(%d) processing in %s' % (train_num + 1, class_path)
            
    print "training data is done"
    writer.close()
    
    
    print "convert val data"
    val_cwd = './data/val_data/'
    val_num = 0
    path_to_val_tfrecords_file=val_cwd+val_ftrecordfilename
    assert not os.path.exists(path_to_val_tfrecords_file), 'The file %s already exists' % path_to_val_tfrecords_file
    writer= tf.python_io.TFRecordWriter(path_to_val_tfrecords_file)
    for index,name in enumerate(classes):
#        print(index)
#        print(name)
        class_path=val_cwd+name+'/'
        for img_name in os.listdir(class_path): 
            val_num=val_num+1
            img_path = class_path+img_name 
            img=Image.open(img_path,'r')
            img = img.resize((28,28))
            img_raw = np.array(img).tobytes()
            example = tf.train.Example(
                 features=tf.train.Features(feature={
                'digit': tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            })) 
            writer.write(example.SerializeToString())
            print '(%d) processing in %s' % (val_num + 1, class_path)
    

    print "val data is done"
    writer.close()

    print "save meta"
    path_to_tfrecords_meta_file = './data/'+meta_filename
    create_tfrecords_meta_file(train_num,val_num, val_num,path_to_tfrecords_meta_file)
    
    print "all done sucessfully!"


if __name__ == '__main__':
    tf.app.run(main=main)