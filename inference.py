import tensorflow as tf
from model import model
import time
from PIL import Image
import os 
import numpy as np 

tf.app.flags.DEFINE_string('image', './image/', 'Path to image file')
tf.app.flags.DEFINE_string('restore_checkpoint', './logs/train/latest.ckpt',
                           'Path to restore checkpoint (without postfix), e.g. ./logs/train/model.ckpt-100')
FLAGS = tf.app.flags.FLAGS

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def main(_):
    path_to_image_file = FLAGS.image
    path_to_restore_checkpoint_file = FLAGS.restore_checkpoint
    img_num = 1
    images = 0
    
    
    for img_name in os.listdir(path_to_image_file):
        img_path = path_to_image_file+img_name 
        img=Image.open(img_path,'r')
#        print img.size
#        image = tf.image.decode_jpeg(tf.read_file(path_to_image_file), channels=1)
        
        img = img.resize((28,28))
        image = tf.reshape(img, [1,28, 28, 1])
        image = tf.image.convert_image_dtype(image, dtype=tf.int32)
#        print image.shape
        if img_num==1:
            images = image
        else:
            print images.shape
            images = tf.concat([images,image],0)
        img_num = img_num +1
#        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

#        image = tf.reshape(image, [1,28, 28, 1])
    images = tf.cast(images,tf.float32)*(1.0/255)-0.5
    images = tf.multiply(tf.subtract(images, 0.5), 2) 
    print images.shape
#    images.reshape(-1,28,28,1)
 
#    image = tf.image.decode_bmp(tf.read_file(path_to_image_file), channels=3)
##    image = tf.image.decode_jpeg(tf.read_file(path_to_image_file), channels=1)
#    image = tf.image.resize_images(image, [54, 54])
#    image = tf.reshape(image, [54, 54, 3])
#    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
#    image = tf.multiply(tf.subtract(image, 0.5), 2)
##    image = tf.image.resize_images(image, [54, 54])
#    images = tf.reshape(image, [1, 54, 54, 3])
    
    X = tf.placeholder("float", [None, 28, 28, 1])
    Y = tf.placeholder("float", [None, 11])
    
    w = init_weights([3, 3, 1, 32])       # 3x3x1 conv, 32 outputs
    w2 = init_weights([3, 3, 32, 64])     # 3x3x32 conv, 64 outputs
    w3 = init_weights([3, 3, 64, 128])    # 3x3x32 conv, 128 outputs
    w4 = init_weights([128 * 4 * 4, 625]) # FC 128 * 4 * 4 inputs, 625 outputs
    w_o = init_weights([625, 11])         # FC 625 inputs, 11 outputs (labels)
       
    p_keep_conv = tf.placeholder("float")
    p_keep_hidden = tf.placeholder("float")
    py_x = model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)
    
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
    train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(loss)
    predict_op = tf.argmax(py_x, 1)



    with tf.Session() as sess:
        restorer = tf.train.Saver()
        restorer.restore(sess, path_to_restore_checkpoint_file)

        start_time = time.time()   
        images= sess.run(images)
        images = images.reshape(-1, 28, 28, 1)  # 28x28x1 input img
#        length_predictions_val, digits_predictions_string_val = sess.run([length_predictions, digits_predictions])
        digits_prediction_val = sess.run(predict_op, feed_dict={X: images,p_keep_conv: 1.0,p_keep_hidden: 1.0})
        duration = time.time() - start_time
        print 'time cost: %f s ' % duration
        
        print digits_prediction_val
#        digits_prediction_val = digits_prediction_val[0]
##        digits_prediction_string_val = digits_predictions_string_val[0]
#        print 'digits:'
#        for index, digit in enumerate(digits_prediction_val):
#            if digit == 10:
#                print 'A'
#            elif digit == 11:
#                None
#            else:
#                print digit
                        
                
                
#        print 'digits: %d' % digits_prediction_val[0]
#        print 'digits: %d' % digits_prediction_val[1]
#        print 'digits: %d' % digits_prediction_val[2]
#        print 'digits: %d' % digits_prediction_val[3]
#        print 'digits: %s' % digits_prediction_string_val[0]
#        print 'digits: %s' % digits_prediction_string_val[1]
#        print 'digits: %s' % digits_prediction_string_val[2]
#        print 'digits: %s' % digits_prediction_string_val[3]
#        print 'digits: %s' % digits_prediction_string_val[4]


if __name__ == '__main__':
    tf.app.run(main=main)