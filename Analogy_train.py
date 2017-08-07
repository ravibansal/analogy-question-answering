'''
Deep Learning Programming Assignment 2
--------------------------------------
Name: Ravi Bansal
Roll No.: 13CS30026

======================================
Complete the functions in this file.
Note: Do not change the function signatures of the train
and test functions
'''
import numpy as np
import tensorflow as tf
import os 
import shutil


def train(trainX):
    input_dim = 1200
    sess = tf.InteractiveSession()
    hidden_dim = 120
    x1 = tf.placeholder(tf.float32,[None,1200])
    x2 = tf.placeholder(tf.float32,[None,1200])
    x3 = tf.placeholder(tf.float32,[None,1200])
    x4 = tf.placeholder(tf.float32,[None,1200])
    x5 = tf.placeholder(tf.float32,[None,1200])

    W1 = tf.Variable(tf.random_uniform([1200,hidden_dim],-(1.0/1200)**(1/2.0),(1.0/1200)**(1/2.0),tf.float32))
    b1 = tf.Variable(tf.zeros([1,hidden_dim]))
    W2 = tf.Variable(tf.random_uniform([hidden_dim,1],-(1.0/hidden_dim)**(1/2.0),(1.0/hidden_dim)**(1.0/2),tf.float32))
    b2 = tf.Variable(tf.zeros([1,1]))

    Z1 = tf.matmul(x1,W1) + b1
    Z2 = tf.matmul(x2,W1) + b1
    Z3 = tf.matmul(x3,W1) + b1
    Z4 = tf.matmul(x4,W1) + b1
    Z5 = tf.matmul(x5,W1) + b1

    s1 = tf.matmul(tf.nn.relu6(Z1),W2) + b2
    s2 = tf.matmul(tf.nn.relu6(Z2),W2) + b2
    s3 = tf.matmul(tf.nn.relu6(Z3),W2) + b2
    s4 = tf.matmul(tf.nn.relu6(Z4),W2) + b2
    s5 = tf.matmul(tf.nn.relu6(Z5),W2) + b2

    sc = tf.maximum(s2,tf.maximum(s3,tf.maximum(s4,s5)))
    s = s1
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_constant = 0.001
    loss = tf.maximum(0.0,1.0+1.0*sc-1.0*s) + reg_constant * sum(reg_losses)
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    sess.run(tf.global_variables_initializer())

    for k in xrange(15):
        perm = np.random.permutation(trainX.shape[0])
        trainX = trainX[perm]
        print "Epoch %d " %(k+1)
        batch_size = 1000
        for i in xrange(0,trainX.shape[0],batch_size):
            x_a1 = trainX[i:min(i+batch_size,trainX.shape[0])]
            x_a1 = x_a1[:,0]
            x_a2 = trainX[i:min(i+batch_size,trainX.shape[0])]
            x_a2 = x_a2[:,1]
            x_a3 = trainX[i:min(i+batch_size,trainX.shape[0])]
            x_a3 = x_a3[:,2]
            x_a4 = trainX[i:min(i+batch_size,trainX.shape[0])]
            x_a4 = x_a4[:,3]
            x_a5 = trainX[i:min(i+batch_size,trainX.shape[0])]
            x_a5 = x_a5[:,4]
            train_step.run(feed_dict={x1:x_a1,x2:x_a2,x3:x_a3,x4:x_a4,x5:x_a5})
    saver = tf.train.Saver()
    tf.add_to_collection('s1', s1)
    tf.add_to_collection('s2', s2)
    tf.add_to_collection('s3', s3)
    tf.add_to_collection('s4', s4)
    tf.add_to_collection('s5', s5)
    tf.add_to_collection('x1', x1)
    tf.add_to_collection('x2', x2)
    tf.add_to_collection('x3', x3)
    tf.add_to_collection('x4', x4)
    tf.add_to_collection('x5', x5)
    if not os.path.exists('saved_model'):
        os.makedirs('saved_model')
    else:
        shutil.rmtree('saved_model')
        os.makedirs('saved_model')
    saver.save(sess, 'saved_model/trained-model')
    sess.close()
    tf.reset_default_graph()

def test(testX, to_return = 0):
    input_dim = 1200
    sess = tf.Session()
    new_saver = tf.train.import_meta_graph('saved_model/trained-model.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./saved_model/'))
    s1 = tf.get_collection('s1')[0]
    s2 = tf.get_collection('s2')[0]
    s3 = tf.get_collection('s3')[0]
    s4 = tf.get_collection('s4')[0]
    s5 = tf.get_collection('s5')[0]
    x1 = tf.get_collection('x1')[0]
    x2 = tf.get_collection('x2')[0]
    x3 = tf.get_collection('x3')[0]
    x4 = tf.get_collection('x4')[0]
    x5 = tf.get_collection('x5')[0]
    num_inputs=testX.shape[0]
    x_1 = testX[:,0].reshape(-1,input_dim)
    x_2 = testX[:,1].reshape(-1,input_dim)
    x_3 = testX[:,2].reshape(-1,input_dim)
    x_4 = testX[:,3].reshape(-1,input_dim)
    x_5 = testX[:,4].reshape(-1,input_dim)
    s_1,s_2,s_3,s_4,s_5 = sess.run([s1,s2,s3,s4,s5],feed_dict={x1:x_1,x2:x_2,x3:x_3,x4:x_4,x5:x_5})
    y_ = np.argmax(np.array(zip(s_1,s_2,s_3,s_4,s_5)),axis=1)
    count = y_.shape[0]-np.count_nonzero(y_)
    sess.close()
    tf.reset_default_graph()
    if to_return == 0:
        return (count*100.0/num_inputs)
    else:
        return (count*100.0/num_inputs), y_

def test_for_train(testX, w2, b2, w1, b1, sess, s1, s2, s3, s4, s5, x1, x2, x3, x4, x5):
    input_dim = 1200
    num_inputs=testX.shape[0]
    # count = 0
    x_1 = testX[:,0].reshape(-1,input_dim)
    x_2 = testX[:,1].reshape(-1,input_dim)
    x_3 = testX[:,2].reshape(-1,input_dim)
    x_4 = testX[:,3].reshape(-1,input_dim)
    x_5 = testX[:,4].reshape(-1,input_dim)
    s_1,s_2,s_3,s_4,s_5 = sess.run([s1,s2,s3,s4,s5],feed_dict={x1:x_1,x2:x_2,x3:x_3,x4:x_4,x5:x_5})
    y_ = np.argmax(np.array(zip(s_1,s_2,s_3,s_4,s_5)),axis=1)
    count = y_.shape[0]-np.count_nonzero(y_)
    return (count*100.0/num_inputs)
