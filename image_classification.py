import ssl
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

ssl._create_default_https_context = ssl._create_unverified_context



mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)



#(Here None means that a dimension can be of any length.)
x = tf.placeholder(tf.float32, [None, 784])

'''
Notice that W has a shape of [784, 10] because we want to multiply the 784-dimensional image vectors
by it to produce 10-dimensional vectors of evidence for the difference classes. 
b has a shape of [10] so we can add it to the output.
'''
#initiate W and b as a tensor full of zeros
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))


#softmax regression model
y = tf.nn.softmax(tf.matmul(x, W) + b)

#cross entropy
y_ = tf.placeholder(tf.float32, [None, 10])
#training step
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

#we ask TensorFlow to minimize cross_entropy using the gradient descent algorithm with a learning rate of 0.5.
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#launch model
sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

#Let's train -- we'll run the training step 1000 times!
'''
Each step of the loop, we get a "batch" of one hundred random data points from our training set. 
We run train_step feeding in the batches data to replace the placeholders.
'''
for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
'''
tf.argmax(y,1) is the label our model thinks is most likely for each input, 
while tf.argmax(y_,1) is the correct label
'''

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

#cast booleans as 0,1 to compute accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

