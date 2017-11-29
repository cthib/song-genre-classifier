import collections
import csv
import ssl
import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelBinarizer
from tensorflow.python.platform import gfile

from genres import Genres


Dataset = collections.namedtuple('Dataset', ['data', 'target'])

def load_csv_with_header(filename, target_dtype, features_dtype, target_column=-1):
	"""Modified from Tensorflow. Load dataset from CSV file with a header row."""
	with gfile.Open(filename) as csv_file:
		data_file = csv.reader(csv_file)
		header = data_file.__next__()
		data, target = [], []
		for row in data_file:
			target.append(row.pop(target_column))
			data.append(np.asarray(row, dtype=features_dtype))
	
	target = np.array(target, dtype=target_dtype)
	data = np.array(data)
	return Dataset(data=data, target=target)


def get_tf_inputs(dataset):
    return tf.estimator.inputs.numpy_input_fn(x={"x": np.array(dataset.data)},
    										  y=np.array(dataset.target),
    										  num_epochs=None,
    									      shuffle=True)


def genre_val(classification):
	inv_genres = Genres().get_inv_genres()
	for i, val in enumerate(classification):
		if val == 1:
			return inv_genres[i] 
	return "Genre not found."


def train_softmax(num_classes=13, num_features=35, save_sess=True, features=[]):
	"""Train softmax classifier on determined number of classes and features."""
	print("Training softmax genre classifier on {} classes and {} features...".format(num_classes, num_features))
	label_binarizer = LabelBinarizer()
	sess = tf.Session()

	training_set_file = 'dataset/balanced' + str(num_classes) + '/train_set.csv'
	test_set_file = 'dataset/balanced' + str(num_classes) + '/test_set.csv'
	feature_cols = [i for i in range(0, num_features)]

	# Extract the train and test data, split into features and labels
	print("- Extracting train and test sets")
	train_df_data = pd.read_csv(training_set_file, usecols=feature_cols)
	train_df_labels = pd.read_csv(training_set_file, usecols=[num_features])
	test_df_data = pd.read_csv(test_set_file,  usecols=feature_cols)
	test_df_labels = pd.read_csv(test_set_file, usecols=[num_features])

	# One hot encoded
	print("- Label binarizing")
	label_binarizer.fit(range(num_classes))
	train_one_hot_encoded= label_binarizer.transform(train_df_labels)
	label_binarizer.fit(range(num_classes))
	test_one_hot_encoded = label_binarizer.transform(test_df_labels)

	# Softmax pre-setup
	Xp = tf.placeholder(tf.float32, shape=[None, num_features], name='x')
	Yp = tf.placeholder(tf.float32, shape=[None, num_classes], name='y')
	W = tf.Variable(tf.truncated_normal([num_features, num_classes], mean=0.0, stddev=0.01))
	b = tf.Variable(tf.truncated_normal([num_classes], mean=0.0, stddev=0.01))

	print("- Softmax setup")
	y_ = tf.matmul(Xp, W) + b
	y = tf.nn.softmax(y_)

	# Reduce mean and optimize
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=Yp))
	tf.reduce_mean(cross_entropy)
	optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

	# Accuracy of the model
	print("- Setting up predictions")
	corr_pred = tf.equal(tf.round(y), tf.round(Yp))
	corr_pred = tf.equal(tf.round(y), tf.round(Yp))
	accuracy = tf.reduce_mean(tf.cast(corr_pred, tf.float32)) * 100
	
	# Run
	sess.run(tf.global_variables_initializer())

	# Steps
	for i in range(10000):    
		sess.run(optimizer, feed_dict={Xp: train_df_data, Yp: train_one_hot_encoded})
	
	accuracy_score = sess.run(accuracy, feed_dict={Xp: test_df_data, Yp: test_one_hot_encoded})
	print('\nAccuracy score: {:0.2f}%\n'.format(accuracy_score))

	# Save the session for later use
	if save_sess:
		sess.run(tf.global_variables_initializer())
		save_path = 'tensors/softmax_balanced' + str(num_classes) + ".ckpt"
		saved = tf.train.Saver(sess, save_path)
		print("Model saved in file: {}".format(saved))

	if len(features):
		feat = np.array([features])
		predictions = sess.run(y, feed_dict={Xp: feat})
		print("The song has been predicted as: {}".format(genre_val(predictions[0])))
	

def classify_softmax(features, num_classes=13, num_features=35):
	"""Classify a feature vector using an existing softmax regression model"""
	saver = tf.train.Saver()
	save_path = 'tensors/softmax_balanced' + str(num_classes) + ".ckpt"
	feat = np.array([features])

	with tf.Session() as sess:
		saver.restore(sess, save_path)
		predictions = sess.run(y, feed_dict={Xp: pred})
		print("The song has been predicted as: {}".format(genre_val(predictions[0])))


def train_dnn(num_classes=13, num_features=35, save_sess=True):
	"""Train deep neural net classifier on determined number of classes and features."""
	print("Training softmax genre classifier on {} classes and {} features...".format(num_classes, num_features))
	training_set_file = 'dataset/balanced' + str(num_classes) + '/train_set.csv'
	test_set_file = 'dataset/balanced' + str(num_classes) + '/test_set.csv'

	print("- Extracting train and test sets")
	train = load_csv_with_header(filename=training_set_file, target_dtype=np.int, features_dtype=np.float32)
	test = load_csv_with_header(filename=test_set_file, target_dtype=np.int, features_dtype=np.float32)

	feature_columns = [tf.feature_column.numeric_column("x", shape=[num_features])]

	# Build 5 layer DNN with [N, 2N, 4N, 8N, 16N] hidden layers
	print("- Building Deep Neural Net")
	classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
												hidden_units=[10, 20, 10],
												n_classes=num_classes,
												model_dir="/tmp/dnn" + str(num_classes))

	# Fit model.
	print("- Fit model")
	classifier.fit(input_fn=get_tf_inputs(train), steps=2000)

	# Evaluate accuracy.
	accuracy_score = classifier.evaluate(input_fn=get_tf_inputs(test))["accuracy"]
	print('\nAccuracy score: {:0.2f}%\n'.format(accuracy_score))

	# Save the session for later use
	if save_sess:
		sess.run(tf.global_variables_initializer())
		save_path = 'tensors/dnn_balanced' + str(num_classes) + ".ckpt"
		saved = tf.train.Saver(sess, save_path)
		print("Model saved in file: {}".format(saved))
