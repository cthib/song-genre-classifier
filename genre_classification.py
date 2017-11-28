import ssl
import sklearn.preprocessing

import tensorflow as tf
import pandas as pd
import numpy as np


def train_softmax(num_classes=13, num_features=35):
	label_binarizer = sklearn.preprocessing.LabelBinarizer()
	sess = tf.InteractiveSession()

	training_set_file = 'dataset/balanced' + str(num_classes) + '/train_set.csv'
	test_set_file = 'dataset/balanced' + str(num_classes) + '/test_set.csv'

	feature_cols = [i for i in range(0, num_features)]

	# Extract the train and test data, split into features and labels
	train_df_data = pd.read_csv(training_set_file, usecols=feature_cols)
	train_df_labels = pd.read_csv(training_set_file, usecols=[num_features])
	
	test_df_data = pd.read_csv(test_set_file,  usecols=feature_cols)
	test_df_labels = pd.read_csv(test_set_file, usecols=[num_features])

	# One hot encoded
	label_binarizer.fit(range(np.max(train_df_labels)+1))
	train_one_hot_encoded= label_binarizer.transform(train_df_labels)

	label_binarizer.fit(range(np.max(test_df_labels)+1))
	test_one_hot_encoded = label_binarizer.transform(test_df_labels)

	print('train labels', train_one_hot_encoded.shape)
	print('testlabels', test_one_hot_encoded.shape)

	Xp = tf.placeholder(tf.float32, shape=[None, num_features], name='x')
	Yp = tf.placeholder(tf.float32, shape=[None, num_classes], name='y')
	
	W = tf.Variable(tf.truncated_normal([num_features, num_classes], mean=0.0, stddev=0.01))
	b = tf.Variable(tf.truncated_normal([num_classes], mean=0.0, stddev=0.01))

	# TODO: Ask Jake what this does
	y_ = tf.matmul(Xp, W) + b
	y = tf.nn.softmax(y_)

	# TODO: Ask Jake what this does
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=Yp))
	tf.reduce_mean(cross_entropy)
	tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

	# Accuracy of the model
	corr_pred = tf.equal(tf.round(y), tf.round(Yp))
	corr_pred = tf.equal(tf.round(y), tf.round(Yp))
	accuracy = tf.reduce_mean(tf.cast(corr_pred, tf.float32)) * 100
	print("Model accuracy: " + str(accuracy))
	
	sess.run(tf.global_variables_initializer())

	for i in range(10000):    
		sess.run(optimizer, feed_dict={Xp:train_df_data, Yp:train_one_hot_encoded})
	

	testRatio = sess.run(accuracy, feed_dict={Xp:test_df_data, Yp:test_one_hot_encoded})
	print('Test accuracy:', testRatio)

if __name__ == "__main__":
	train_softmax()