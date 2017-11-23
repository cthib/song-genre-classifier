import ssl
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
ssl._create_default_https_context = ssl._create_unverified_context


#read data (features for each track)
df = pd.read_csv('dataset/feature_subset.csv', usecols = range(1, 36), skiprows = [0],  header=None, nrows=100)
d = df.values


#read labels
l = pd.read_csv('dataset/genre_subset.csv', usecols = [1], skiprows= [0], header=None, nrows=100)
labels = l.values


#35 features
print(len(df.columns))
#13 genres
print(l[1].unique())
'''
#convert to strings
data = np.float64(d)
labels = np.array(l,'str')


print(data)
print(labels)

x = tf.placeholder(tf.float64, shape=[100, 35])
x = data
w = tf.random_normal([35,100],mean=0.0, stddev=1.0, dtype=tf.float64)
y = tf.nn.softmax(tf.matmul(x,w))


#cross entropy
y_ = tf.placeholder(tf.float64, [None, 13])

#training step
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

#we ask TensorFlow to minimize cross_entropy using the gradient descent algorithm with a learning rate of 0.5.
#train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#launch model
sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

#Let's train -- we'll run the training step 1000 times!

#Each step of the loop, we get a "batch" of one hundred random data points from our training set. 
#We run train_step feeding in the batches data to replace the placeholders.

with tf.Session() as sess:
  
  print (sess.run(y))

#tf.argmax(y,1) is the label our model thinks is most likely for each input, 
#while tf.argmax(y_,1) is the correct label


correct_predictions = tf.equal(tf.argmax(y, 1), tf.cast(labels, tf.int64))
accuracy_op = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

#cast booleans as 0,1 to compute accuracy
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))
#print(sess.run(accuracy, feed_dict={x: data, y_: labels}))
accuracy = sess.run([accuracy_op], feed_dict={x: data, y_: labels})
print(accuracy)
'''