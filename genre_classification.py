import ssl
import tensorflow as tf
import pandas as pd
import numpy as np


def train_base():
    """
    Training of genres from FMA dataset.
    """
    ssl._create_default_https_context = ssl._create_unverified_context

    #read data (features for each track)
    df = pd.read_csv('dataset/feature_subset.csv', usecols = range(1, 36), skiprows = [0],  header=None)
    d = df.values


    #read labels
    l = pd.read_csv('dataset/genre_subset.csv', usecols = [1], skiprows = [0], header=None)
    labels = l.values


    #35 features
    print(len(df.columns))
    #13 genres
    print(len(l[1].unique()))

    #convert to strings
    data = np.float32(d)
    labels = np.array(l, 'str')

    print(data)
    print(labels)

    x = tf.placeholder(tf.float32, shape=(48598, 35))
    x = data

    W = tf.Variable(tf.zeros([35, 13]))
    y = tf.nn.softmax(tf.matmul(x,W))


    #cross entropy
    y_ = tf.placeholder(tf.float32, [None, 13])

    #training step
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

    #we ask TensorFlow to minimize cross_entropy using the gradient descent algorithm with a learning rate of 0.5.
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    #launch model
    sess = tf.InteractiveSession()

    tf.global_variables_initializer().run()

    #Let's train -- we'll run the training step 1000 times!

    #Each step of the loop, we get a "batch" of one hundred random data points from our training set. 
    #We run train_step feeding in the batches data to replace the placeholders.

    for _ in range(1000):
      batch_xs, batch_ys = mnist.train.next_batch(100)
      sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    #tf.argmax(y,1) is the label our model thinks is most likely for each input, 
    #while tf.argmax(y_,1) is the correct label


    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

    #cast booleans as 0,1 to compute accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

