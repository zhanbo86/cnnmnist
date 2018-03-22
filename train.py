import os
import numpy as np
from datetime import datetime
import time
import tensorflow as tf
from meta import Meta
from donkey import Donkey
from model import model
from PIL import Image
import matplotlib.pyplot as plt
#from evaluator import Evaluator

tf.app.flags.DEFINE_string('data_dir', './data', 'Directory to read TFRecords files')
tf.app.flags.DEFINE_string('train_logdir', './logs/train', 'Directory to write training logs')
tf.app.flags.DEFINE_string('restore_checkpoint', None,
                           'Path to restore checkpoint (without postfix), e.g. ./logs/train/model.ckpt-100')
tf.app.flags.DEFINE_integer('batch_size', 128, 'Default 32')
tf.app.flags.DEFINE_float('learning_rate', 1e-2, 'Default 1e-2')
tf.app.flags.DEFINE_integer('patience', 100, 'Default 100, set -1 to train infinitely')
tf.app.flags.DEFINE_integer('decay_steps', 10000, 'Default 10000')
tf.app.flags.DEFINE_float('decay_rate', 0.9, 'Default 0.9')
tf.app.flags.DEFINE_float('num_epoch', None, 'Default None')
FLAGS = tf.app.flags.FLAGS


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def _train(path_to_train_tfrecords_file,num_train_examples, path_to_val_tfrecords_file, num_val_examples,
           path_to_train_log_dir, path_to_restore_checkpoint_file, training_options):
    batch_size = training_options['batch_size']
    initial_patience = training_options['patience']
    num_steps_to_show_loss = 1
    num_steps_to_check = 1
#    num_steps_to_show_loss = 10
#    num_steps_to_check = 100
#    cwd = './img_confirm/'
    



    with tf.Graph().as_default():
        X = tf.placeholder("float", [None, 28, 28, 1])
        Y = tf.placeholder("float", [None, 11])
        
        w = init_weights([3, 3, 1, 32])       # 3x3x1 conv, 32 outputs
        w2 = init_weights([3, 3, 32, 64])     # 3x3x32 conv, 64 outputs
        w3 = init_weights([3, 3, 64, 128])    # 3x3x32 conv, 128 outputs
        w4 = init_weights([128 * 4 * 4, 625]) # FC 128 * 4 * 4 inputs, 625 outputs
        w_o = init_weights([625, 11])         # FC 625 inputs, 11 outputs (labels)
        
        global_step = tf.Variable(0, name='global_step', trainable=False)
        
        p_keep_conv = tf.placeholder("float")
        p_keep_hidden = tf.placeholder("float")
        py_x = model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)
        
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
        train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(loss,global_step=global_step)
        predict_op = tf.argmax(py_x, 1)

        print num_train_examples
        trX,trY = Donkey.build_batch(path_to_train_tfrecords_file,num_examples=num_train_examples,batch_size=batch_size,
                                     shuffled=True,num_epoch_=training_options['num_epoch'])
        teX,teY = Donkey.build_batch(path_to_val_tfrecords_file,num_examples=num_val_examples,batch_size=64,
                                     shuffled=True,num_epoch_=training_options['num_epoch'])
#        teX,teY = Donkey.build_batch_from_file(path_to_val_file,batch_size=10,
#                                               shuffled=True,num_epoch_=training_options['num_epoch'])
        
        indices = tf.placeholder("uint8",[batch_size,1])
        var = tf.one_hot(indices,depth = 11,axis = 1)
        
        tf.summary.image('image', trX)
        tf.summary.scalar('loss', loss)
        summary = tf.summary.merge_all()
#        
#        
#
#        tr_x_split = tf.unstack(trX,32,axis=0)
#        


        
#        image_batch,length_batch,digits_batch = Donkey.build_batch(path_to_train_tfrecords_file1,path_to_train_tfrecords_file2,path_to_train_tfrecords_file3,path_to_train_tfrecords_file4,
#                                                                     num_examples=num_train_examples,
#                                                                     batch_size=batch_size,
#                                                                    shuffled=True,num_epoch_=training_options['num_epoch'])
#        

#        trX = trX.reshape(-1, 28, 28, 1)  # 28x28x1 input img
#        teX = teX.reshape(-1, 28, 28, 1)  # 28x28x1 input img
        
        with tf.Session() as sess:
            summary_writer = tf.summary.FileWriter(path_to_train_log_dir, sess.graph)
            
            
            # you need to initialize all variables
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            
            saver = tf.train.Saver()
            if path_to_restore_checkpoint_file is not None:
                assert tf.train.checkpoint_exists(path_to_restore_checkpoint_file), \
                    '%s not found' % path_to_restore_checkpoint_file
                saver.restore(sess, path_to_restore_checkpoint_file)
                print 'Model restored from file: %s' % path_to_restore_checkpoint_file


            print 'Start training'
#            print trX.shape
        
            for i in range(20000):
#                plt.imshow(trX.eval())
#                plt.show()
#                single,l = sess.run([trX,trY])
#                single.resize(28,28)
#                img=Image.fromarray(single, 'L')
#                img.save(cwd+str(i)+'_''Label_'+str(l)+'.jpg')
                #print(single,l)

                
#                tr_x,tr_y,tr_x_split = sess.run([trX,trY,tr_x_split])
#                print tr_x.shape
#                tr_x = tr_x.reshape(-1, 28, 28, 1)  # 28x28x1 input img
##                print tr_x.shape,tr_x_split
#
#                
#                for j,ele in enumerate(tr_x_split):
##                    print ele.shape
##                    ele.reshape(28,28)
##                    print ele.shape
#                    ele.resize(28,28)
#                    img = Image.fromarray(ele,'L')
#                    img.save(cwd+str(j)+'_''Label_'+str(tr_y[j])+'.jpg')
##                    print ele,ele.shape,tr_y[j]
#                
                tr_x,tr_y = sess.run([trX,trY])
                tr_y.resize(batch_size,1)
                tr_y = sess.run(var,feed_dict={indices:tr_y})
                tr_y.resize(batch_size,11)
                
#                tr_y = tr_y.reshape(-1,11)        
                _, loss_val, summary_val, global_step_val=sess.run([train_op, loss, summary, global_step], 
                                                                   feed_dict={X: tr_x, Y: tr_y,p_keep_conv: 0.8, p_keep_hidden: 0.5})

                te_x,te_y = sess.run([teX,teY])
                te_x = te_x.reshape(-1, 28, 28, 1)  # 28x28x1 input img
                accuracy = np.mean(te_y ==sess.run(predict_op, feed_dict={X: te_x,p_keep_conv: 1.0,p_keep_hidden: 1.0}))
                print(i, accuracy)
                summary_writer.add_summary(summary_val, global_step=global_step_val)
                path_to_latest_checkpoint_file = saver.save(sess, os.path.join(path_to_train_log_dir, 'latest.ckpt'))


            coord.request_stop()
            coord.join(threads)
            print 'Finished'
        
#        global_step = tf.Variable(0, name='global_step', trainable=False)

        
        
#        length_logtis,digits_logits = Model.inference(image_batch, drop_rate=0.2)
#        loss = Model.loss(length_logtis,digits_logits,length_batch, digits_batch)
#
#        global_step = tf.Variable(0, name='global_step', trainable=False)
#        learning_rate = tf.train.exponential_decay(training_options['learning_rate'], global_step=global_step,
#                                                   decay_steps=training_options['decay_steps'], decay_rate=training_options['decay_rate'], staircase=True)
#        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
#        train_op = optimizer.minimize(loss, global_step=global_step)

#        tf.summary.image('image', image_batch)
#        tf.summary.scalar('loss', loss)
#        summary = tf.summary.merge_all()
        

#        config = tf.ConfigProto()
#        config.gpu_options.per_process_gpu_memory_fraction = 0.6
#        config.gpu_options.allow_growth = True
#        with tf.Session(config=config) as sess:
#        with tf.Session() as sess:
#            summary_writer = tf.summary.FileWriter(path_to_train_log_dir, sess.graph)
#            evaluator = Evaluator(os.path.join(path_to_train_log_dir, 'eval/val'))
#
#            sess.run(tf.global_variables_initializer())
#            sess.run(tf.local_variables_initializer())
#            coord = tf.train.Coordinator()
#            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#
#            saver = tf.train.Saver()
#            if path_to_restore_checkpoint_file is not None:
#                assert tf.train.checkpoint_exists(path_to_restore_checkpoint_file), \
#                    '%s not found' % path_to_restore_checkpoint_file
#                saver.restore(sess, path_to_restore_checkpoint_file)
#                print 'Model restored from file: %s' % path_to_restore_checkpoint_file
#
#            print 'Start training'
#            patience = initial_patience
#            best_accuracy = 0.0
#            duration = 0.0
#
#            while True:
#                start_time = time.time()
#                _, loss_val, summary_val, global_step_val= sess.run([train_op, loss, summary, global_step])
#                duration += time.time() - start_time
#
#                if global_step_val % num_steps_to_show_loss == 0:
#                    examples_per_sec = batch_size * num_steps_to_show_loss / duration
#                    duration = 0.0
#                    print '=> %s: step %d, loss = %f (%.1f examples/sec)' % (
#                        datetime.now(), global_step_val, loss_val, examples_per_sec)
#
#                if global_step_val % num_steps_to_check != 0:
#                    continue
#
#                summary_writer.add_summary(summary_val, global_step=global_step_val)
#
#                print '=> Evaluating on validation dataset...'
#                path_to_latest_checkpoint_file = saver.save(sess, os.path.join(path_to_train_log_dir, 'latest.ckpt'))
#                accuracy = np.mean(np.argmax(teY[test_indices], axis=1) ==sess.run(predict_op, feed_dict={X: teX[test_indices],
#                                                         p_keep_conv: 1.0,
#                                                         p_keep_hidden: 1.0}))
#                print '==> accuracy = %f, best accuracy %f' % (accuracy, best_accuracy)
#
#                if accuracy > best_accuracy:
#                    path_to_checkpoint_file = saver.save(sess, os.path.join(path_to_train_log_dir, 'model.ckpt'),
#                                                         global_step=global_step_val)
#                    print '=> Model saved to file: %s' % path_to_checkpoint_file
#                    patience = initial_patience
#                    best_accuracy = accuracy
#                else:
#                    patience -= 1
#
#                print '=> patience = %d' % patience
#                if patience == 0:
#                    break
#
#            coord.request_stop()
#            coord.join(threads)
#            print 'Finished'


def main(_):
    path_to_train_tfrecords_file = os.path.join(FLAGS.data_dir, 'train_data/train.tfrecords')
    path_to_val_tfrecords_file = os.path.join(FLAGS.data_dir, 'val_data/val.tfrecords')
    path_to_tfrecords_meta_file = os.path.join(FLAGS.data_dir, 'meta.json')
    path_to_train_log_dir = FLAGS.train_logdir
    path_to_restore_checkpoint_file = FLAGS.restore_checkpoint
    training_options = {
        'batch_size': FLAGS.batch_size,
        'learning_rate': FLAGS.learning_rate,
        'patience': FLAGS.patience,
        'decay_steps': FLAGS.decay_steps,
        'decay_rate': FLAGS.decay_rate,
        'num_epoch':FLAGS.num_epoch
    }

    meta = Meta()
    meta.load(path_to_tfrecords_meta_file)

    _train(path_to_train_tfrecords_file,meta.num_train_examples,path_to_val_tfrecords_file,meta.num_val_examples,
           path_to_train_log_dir, path_to_restore_checkpoint_file,
           training_options)
#    _train(path_to_train_tfrecords_file1,path_to_train_tfrecords_file2,path_to_train_tfrecords_file3,path_to_train_tfrecords_file4, 457500,
#           path_to_val_tfrecords_file, 82,
#           path_to_train_log_dir, path_to_restore_checkpoint_file,
#           training_options)
#    _train(path_to_train_tfrecords_file1,path_to_train_tfrecords_file2,path_to_train_tfrecords_file3,path_to_train_tfrecords_file4, 7400,
#           path_to_val_tfrecords_file, 82,
#           path_to_train_log_dir, path_to_restore_checkpoint_file,
#           training_options)

if __name__ == '__main__':
    tf.app.run(main=main)
