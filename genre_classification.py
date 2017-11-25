import ssl
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


sess = tf.InteractiveSession()


cols_train = pd.read_csv('dataset/train_set10.csv', nrows=1).columns
cols_test = pd.read_csv('dataset/test_set10.csv', nrows=1).columns
train_df_data = pd.read_csv('dataset/train_set10.csv', usecols=range(0,10))
train_df_labels = pd.read_csv('dataset/train_set10.csv', usecols=cols_train[-1])

test_df_data = pd.read_csv('dataset/test_set10.csv',  usecols=range(0,10))
test_df_labels = pd.read_csv('dataset/test_set10.csv', usecols=cols_test[-1])


#test_df_labels_onehot = tf.one_hot(test_df_labels, depth=8).eval()
#train_df_labels_onehot = tf.one_hot(train_df_labels, depth=8).eval()


#test_df_data.to_csv('dataset/test_set_data.csv')
#test_df_labels.to_csv('dataset/test_set_labels.csv')

print('Training set', train_df_data.shape)
print('Testing set', train_df_labels.shape)




Xp = tf.placeholder(tf.float32, shape=[None, 10], name='x')
Yp = tf.placeholder(tf.float32, shape=[None,1], name='y')

W = tf.Variable(tf.zeros([10, 8]))
b = tf.Variable(tf.zeros([8]))

#Y_= 

y = tf.nn.softmax(tf.matmul(Xp, W) + b)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(Yp * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


#crossEntropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=Y_, labels=Yp))
#optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(crossEntropy)


##accuracy
correntPredict = tf.equal(tf.arg_max(y, 1), tf.arg_max(Yp, 1))
accuracy = tf.reduce_mean(tf.cast(correntPredict, tf.float32))


### Run a session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(8000):    
    sess.run(train_step, feed_dict={Xp:train_df_data, Yp:train_df_labels})
   

testRatio = sess.run(accuracy, feed_dict={Xp:test_df_data, Yp:test_df_labels})
print('test accuracy:', testRatio)
