import ssl
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


sess = tf.InteractiveSession()

num_features = 35
num_classes = 13

training_set_file = 'dataset/balanced13/train_set.csv'
test_set_file = 'dataset/balanced13/test_set.csv'

cols_train = pd.read_csv(training_set_file, nrows=1).columns
cols_test = pd.read_csv(test_set_file, nrows=1).columns
range1 = [i for i in range(0,35)]
train_df_data = pd.read_csv(training_set_file, usecols=range1)
#train_df_labels = pd.read_csv(training_set_file, usecols=[num_features])

test_df_data = pd.read_csv(test_set_file,  usecols=range1)
#test_df_labels = pd.read_csv(test_set_file, usecols=[num_features])

#map labels to ints
'''
train_df_labels.iloc[:,-1] = train_df_labels.iloc[:,-1].map({'Electronic':0, 'Pop':1, 'Experimental':2, 'Rock':3, 'International':4, 'Hip-Hop':5, 'Folk':6,
	 'Classical':7, 'Instrumental':8, 'Jazz':9, 'Country':10, 'Blues':11, 'Soul-RnB':12})
test_df_labels.iloc[:,-1] = test_df_labels.iloc[:,-1].map({'Electronic':0, 'Pop':1, 'Experimental':2, 'Rock':3, 'International':4, 'Hip-Hop':5, 'Folk':6,
	 'Classical':7, 'Instrumental':8, 'Jazz':9, 'Country':10, 'Blues':11, 'Soul-RnB':12})
'''
#test_df_labels_onehot = tf.one_hot(test_df_labels, depth=8).eval()
#train_df_labels_onehot = tf.one_hot(train_df_labels, depth=8).eval()

'''
train_df_labels.to_csv('dataset/train_set_labels.csv')
test_df_labels.to_csv('dataset/test_set_labels.csv')
'''
train_df_labels = pd.read_csv('dataset/train_set_labels.csv')
test_df_labels = pd.read_csv('dataset/test_set_labels.csv')

train_one_hot_encoded = tf.one_hot(indices=tf.constant(train_df_labels), depth=num_classes).eval()
test_one_hot_encoded = tf.one_hot(indices=tf.constant(test_df_labels), depth=num_classes).eval()




print('Training set features', train_df_data.shape)
print('training labels', train_df_labels.shape)

'''

Xp = tf.placeholder(tf.float32, shape=[None, num_features], name='x')
Yp = tf.placeholder(tf.float32, shape=[None, 1], name='y')
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
W = tf.Variable(tf.zeros([num_features, num_classes]))
b = tf.Variable(tf.zeros([num_classes]))

#Y_= 

y = tf.nn.log_softmax(tf.matmul(Xp, W) + b)

#cross_entropy = tf.reduce_mean(-tf.reduce_sum(Yp * tf.log(y)))

cross_entropy = tf.reduce_mean(-tf.reduce_sum(Yp*y))


train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


#crossEntropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=Y_, labels=Yp))
#optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(crossEntropy)


##accuracy
correctPredict = tf.equal(tf.argmax(y, 1), tf.argmax(Yp, 1))
accuracy = tf.reduce_mean(tf.cast(correctPredict, tf.float32))
print(accuracy)

### Run a session

sess.run(tf.global_variables_initializer())

for i in range(1000):    
    sess.run(train_step, feed_dict={Xp:train_df_data, Yp:train_df_labels, keep_prob: 1.0})
   

testRatio = sess.run(accuracy, feed_dict={Xp:test_df_data, Yp:test_df_labels, keep_prob: 1.0})
print('test accuracy:', testRatio)

pred = [[1398.7202022016106, 434.71777491237276, 0.10965552115861281, 
3.3119555357964057, 1376.2821557605066, 0.0, 4866.288341456403, 1279.2643675958077, 
286.92515115868014, -1.4815406416537815, 8.4593327852187361, 1313.5044361495909, 0.0, 
3341.9721960906522, 2859.3233578889267, 855.01963289733078, -0.41947234294003277, 
2.2556051183909398, 2896.2158203125, 0.0, 8688.6474609375, 2.5273113, 1.228359, 0.32544681, 0.043312073, 2.4202311, 0.0, 7.3073134, 0.086118057914402177, 0.042002276281770334, 1.0796525278857523, 2.7986276477609664, 0.0810546875, 0.0, 0.34814453125]]


predictions = sess.run(y, feed_dict={Xp: pred, keep_prob: 1.0})
print(predictions)
'''