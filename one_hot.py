import tensorflow as tf
import sklearn.preprocessing
import pandas as pd
import numpy as np

label_binarizer = sklearn.preprocessing.LabelBinarizer()


sess = tf.InteractiveSession()

num_features = 35
num_classes = 13
learning_rate = 0.5

int_to_genre_map = {0:'Electronic', 1:'Pop', 2:'Experimental', 3:'Rock', 4:'International', 5:'Hip-Hop', 6:'Folk',
	 7:'Classical', 8:'Instrumental', 9:'Jazz', 10:'Country', 11:'Blues', 12:'Soul-RnB'}

genre_to_int_map = {'Electronic':0, 'Pop':1, 'Experimental':2, 'Rock':3, 'International':4, 'Hip-Hop':5, 'Folk':6,
	 'Classical':7, 'Instrumental':8, 'Jazz':9, 'Country':10, 'Blues':11, 'Soul-RnB':12}



training_set_file = 'dataset/balanced13/train_set.csv'
test_set_file = 'dataset/balanced13/test_set.csv'

column_range = [i for i in range(0,num_features)]

train_df_data = pd.read_csv(training_set_file, usecols=column_range)
#train_df_labels = pd.read_csv(training_set_file, usecols=[num_features])
test_df_data = pd.read_csv(test_set_file,  usecols=column_range)
#test_df_labels = pd.read_csv(test_set_file, usecols=[num_features])


#map labels to ints
'''
train_df_labels.iloc[:,-1] = train_df_labels.iloc[:,-1].map(genre_to_int_map)
test_df_labels.iloc[:,-1] = test_df_labels.iloc[:,-1].map(genre_to_int_map)

train_df_labels.to_csv('dataset/train_set_labels.csv')
test_df_labels.to_csv('dataset/test_set_labels.csv')
'''


train_df_labels = pd.read_csv('dataset/train_set_labels.csv')
test_df_labels = pd.read_csv('dataset/test_set_labels.csv')

#convert labels to one hot representation
train_df_labels = train_df_labels.values
test_df_labels = test_df_labels.values
label_binarizer.fit(range(np.max(train_df_labels)+1))
train_one_hot_encoded= label_binarizer.transform(train_df_labels)
label_binarizer.fit(range(np.max(test_df_labels)+1))
test_one_hot_encoded = label_binarizer.transform(test_df_labels)
train_one_hot_encoded = pd.DataFrame(train_one_hot_encoded)
test_one_hot_encoded  = pd.DataFrame(test_one_hot_encoded)

Xp = tf.placeholder(tf.float32, shape=[None, num_features], name='x')
Yp = tf.placeholder(tf.float32, shape=[None, num_classes], name='y')
W = tf.Variable(tf.truncated_normal([num_features, num_classes], mean=0.0, stddev=0.01))
b = tf.Variable(tf.truncated_normal([num_classes], mean=0.0, stddev=0.01))

y_ = tf.matmul(Xp, W) + b
y = tf.nn.softmax(y_)

# computes the error for softmax.
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=Yp))

#GradientDescentOptimizer minimizes error. 
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)


##accuracy
correctPredict = tf.equal(tf.round(y), tf.round(Yp))
accuracy = tf.reduce_mean(tf.cast(correctPredict, tf.float32))*100

### Run a session
sess.run(tf.global_variables_initializer())

for i in range(25000):    
    sess.run(optimizer, feed_dict={Xp:train_df_data, Yp:train_one_hot_encoded})
   

testRatio = sess.run(accuracy, feed_dict={Xp:test_df_data, Yp:test_one_hot_encoded})
print('test accuracy:', testRatio)


# classify input
pred = np.array([[1318.2869097990506, 673.85979913651761, 1.6521337497341229, 3.4056791186110154, 1136.4836273362353, 0.0, 5511.8843703726525, 1593.5475041273635, 442.18441858102659, 0.18337386741734729, 0.15867119426527276, 1529.5494392022711, 0.0, 3581.8049319360712, 2800.2553718427753, 1471.0506074576508, 0.6546489028464868, -0.44537851706137888, 2519.384765625, 0.0, 9431.54296875, 1.7678047, 0.68314832, 0.050609998, 0.50379992, 1.7595094, 0.0, 6.1436887, 0.053003386038080226, 0.045303279550192435, 3.5771099709365806, 17.278182906679763, 0.041015625, 0.0, 0.4501953125]])
predictions = sess.run(y, feed_dict={Xp: pred})
classification = predictions[0]

#convert int to genre
for i in range(0, num_classes):
	if classification[i] == 1.:
		print("The input song has been classified as: {}".format(int_to_genre_map[i]))


		


