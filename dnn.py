from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.contrib import learn


import os
import urllib
import pandas as pd
import numpy as np
import tensorflow as tf

genre8_dict = {
    "Rock": 0,
    "Experimental": 1,
    "Electronic": 2,
    "Hip-Hop": 3,
    "Folk": 4,
    "Pop": 5,
    "Instrumental": 6,
    "International": 7,
}
#load into pandas dataframe, convert genres to ints, save back to csv.
def load_datasets():
	
	train_df = pd.read_csv('dataset/balanced8/train_set.csv')
	test_df = pd.read_csv('dataset/balanced8/test_set.csv')


	#map 8 genres
	train_df.iloc[:,-1] = train_df.iloc[:,-1].map({
    "Rock": 0,
    "Experimental": 1,
    "Electronic": 2,
    "Hip-Hop": 3,
    "Folk": 4,
    "Pop": 5,
    "Instrumental": 6,
    "International": 7
})
	test_df.iloc[:,-1] = test_df.iloc[:,-1].map({
    "Rock": 0,
    "Experimental": 1,
    "Electronic": 2,
    "Hip-Hop": 3,
    "Folk": 4,
    "Pop": 5,
    "Instrumental": 6,
    "International": 7
})
	#map 13 genres
	'''
	train_df.iloc[:,-1] = train_df.iloc[:,-1].map({'Electronic':0, 'Pop':1, 'Experimental':2, 'Rock':3, 'International':4, 'Hip-Hop':5, 'Folk':6,
	 'Classical':7, 'Instrumental':8, 'Jazz':9, 'Country':10, 'Blues':11, 'Soul-RnB':12})
	test_df.iloc[:,-1] = test_df.iloc[:,-1].map({'Electronic':0, 'Pop':1, 'Experimental':2, 'Rock':3, 'International':4, 'Hip-Hop':5, 'Folk':6,
	 'Classical':7, 'Instrumental':8, 'Jazz':9, 'Country':10, 'Blues':11, 'Soul-RnB':12})
	'''
	#print(len(test_df.columns))
	#print(test_df['genre_top'].unique())
	#print(train_df.iloc[:,-1].unique())

	train_df.to_csv('dataset/train_set.csv')
	test_df.to_csv('dataset/test_set.csv')

load_datasets()

# Data sets
TRAIN_SET = "dataset/train_set10.csv"
TEST_SET = "dataset/test_set10.csv"

train_df = pd.read_csv('dataset/test_set10.csv')

test_df = pd.read_csv('dataset/test_set10.csv')


print('Training set', train_df.shape)
print('Testing set', test_df.shape)



print(len(test_df.columns))
print(test_df.iloc[:,-1].unique())


train = tf.contrib.learn.datasets.base.load_csv_without_header(
      filename=TRAIN_SET,
      target_dtype=np.int,
      features_dtype=np.float32)

test = tf.contrib.learn.datasets.base.load_csv_without_header(
      filename=TEST_SET,
      target_dtype=np.int,
      features_dtype=np.float32)

feature_columns = learn.infer_real_valued_columns_from_input(train.data)

# Build 3 layer DNN with 10, 20, 10 units respectively. n,2n,4n,8n,16n
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                              hidden_units=[4, 8, 2*8, 4*8, 8*8],
                                              n_classes=8,
                                              model_dir="/tmp/genre_model7")
# Define the training input2
def get_train_inputs():
    x = tf.constant(train.data)
    y = tf.constant(train.target)

    return x, y

  # Fit model.
classifier.fit(input_fn=get_train_inputs, steps=2000)
# Define the test inputs
def get_test_inputs():
    x = tf.constant(test.data)
    y = tf.constant(test.target)

    return x, y

  # Evaluate accuracy.
accuracy_score = classifier.evaluate(input_fn=get_train_inputs,
                                       steps=1)["accuracy"]
print("\nTest Accuracy: {0:f}\n".format(accuracy_score))


#predictions = list(classifier.predict(input_fn=new_samples))
