import tensorflow as tf
from model import model
from PIL import Image
import time
from PIL import Image
import os 
import numpy as np 
import cv2 as cv
import ctypes
import stringSegment as stringSeg
from PyQt4.Qt import *
from PyQt4.QtCore import * 
from PyQt4.QtGui import *  
import sys  


#so = ctypes.cdll.LoadLibrary
#stringSeg = so("/home/zb/BoZhan/ocr_ws/devel/lib/libstrSeg.so")
#from libstrSeg import getChars

class InferenceMain(QThread): 
    def __init__(self,parent=None):  
        tf.app.flags.DEFINE_string('image', '/home/zb/BoZhan/ocr_ws/src/easyocr/rec_char_img/', 'Path to image file')
        tf.app.flags.DEFINE_string('restore_checkpoint', './logs/train/latest.ckpt',
                           'Path to restore checkpoint (without postfix), e.g. ./logs/train/model.ckpt-100')
        self.FLAGS = tf.app.flags.FLAGS
        self.digits_prediction_val = 0
        self.duration = 0
        self.do = False


    def init_weights(self,shape):
        return tf.Variable(tf.random_normal(shape, stddev=0.01))
    
    def do_inference(self):
        self.do = True
    
    
    def startRun(self):
#        path_to_image_file = self.FLAGS.image
        path_to_restore_checkpoint_file = self.FLAGS.restore_checkpoint
        img_num = 1
        images = 0
#        
#        files = os.listdir(path_to_image_file)
#        files.sort()
#        
#        img_name = files[0]
#        img_path = path_to_image_file+img_name
#        cvimg = cv.imread(img_path,0)
#        img = stringSeg.Mat.from_array(cvimg)
#        charVec = stringSeg.matVector()
#        charVec = stringSeg.getCharsMain(img)
        charVec = stringSeg.matVector()
        charVec = stringSeg.getCharsMain()
        print("the length of charVec is %d"%len(charVec))
        
#        wholeImg = np.asarray(charVec[0])
#        wholeImg = Image.fromarray(wholeImg)
#        wholeImg = wholeImg.resize(100,100)
              
        for index in range(1,len(charVec),1):
            img = np.asarray(charVec[index])
            img = Image.fromarray(img)
            img = img.resize((28,28))
    #        cv.imshow('charVec',img)
    #        cv.waitKey(10000)
            
            image = tf.reshape(img, [1,28, 28, 1])
            image = tf.image.convert_image_dtype(image, dtype=tf.int32)
            if img_num==1:
                images = image
            else:
    #            print images.shape
                images = tf.concat([images,image],0)
            img_num = img_num +1
    #        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    
    #        image = tf.reshape(image, [1,28, 28, 1])
        images = tf.cast(images,tf.float32)*(1.0/255)-0.5
        images = tf.multiply(tf.subtract(images, 0.5), 2) 
        print images.shape
    #    images.reshape(-1,28,28,1)
     
        
        X = tf.placeholder("float", [None, 28, 28, 1])
        Y = tf.placeholder("float", [None, 11])
        
        w = self.init_weights([3, 3, 1, 32])       # 3x3x1 conv, 32 outputs
        w2 = self.init_weights([3, 3, 32, 64])     # 3x3x32 conv, 64 outputs
        w3 = self.init_weights([3, 3, 64, 128])    # 3x3x32 conv, 128 outputs
        w4 = self.init_weights([128 * 4 * 4, 625]) # FC 128 * 4 * 4 inputs, 625 outputs
        w_o = self.init_weights([625, 11])         # FC 625 inputs, 11 outputs (labels)
           
        p_keep_conv = tf.placeholder("float")
        p_keep_hidden = tf.placeholder("float")
        py_x = model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)
        
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
        train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(loss)
        predict_op = tf.argmax(py_x, 1)
    
    
    
        with tf.Session() as sess:
            restorer = tf.train.Saver()
            restorer.restore(sess, path_to_restore_checkpoint_file)
            print "start inference:"
            

            start_time = time.time()   
            images= sess.run(images)
            images = images.reshape(-1, 28, 28, 1)  # 28x28x1 input img
            #        length_predictions_val, digits_predictions_string_val = sess.run([length_predictions, digits_predictions])
            self.digits_prediction_val = sess.run(predict_op, feed_dict={X: images,p_keep_conv: 1.0,p_keep_hidden: 1.0})
            self.duration = time.time() - start_time
            print 'time cost: %f s ' % self.duration         
            print self.digits_prediction_val
#            sess.close()

            

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
    
    def stop(self):
        print("stop")

#if __name__ == '__main__':
#    tf.app.run(main=main)