import Audio_data
import Labels
import numpy as np
import tensorflow as tf
import os
import random
import sys
import matplotlib.pyplot as plt
import datetime


#dir of data and labels
dir_data="./data/Datas/"
dir_label="./data/Labels/"
dir_test_data="./data/Test_datas/"
dir_test_label="./data/Test_labels/"
dir_cross_data="./data/Cross_datas/"
dir_cross_label="./data/Cross_labels/"
# dir_test_data="./data/Cross_datas/"
# dir_test_label="./data/Cross_labels/"
# directory = '/home/datalab/Documents/Data/'
directory = '/home/datalab/Documents/NormalizedData/'
save_path_latest="./Alg6_1channel/RNN_MultiLayer.ckpt"


#####################################

root_dir = []
for root_d, dirs_d, files_d in os.walk(directory):
    root_dir.append(root_d)

root_dir.remove(directory)
random.shuffle(root_dir)
print root_dir
print len(root_dir)


training_dir = root_dir[:25]
cross_dir = root_dir[25:30]
testing_dir = root_dir[30:37]


training_files_d = []
training_files_l = []

for row in training_dir:
    for root_d, dirs_d, files_d in os.walk(row):
        if 'csv' in files_d[0]:
            training_files_l.append(root_d+"/"+files_d[0])
            training_files_d.append(root_d+"/"+files_d[1])
        else:
            training_files_d.append(root_d+"/"+files_d[0])
            training_files_l.append(root_d+"/"+files_d[1])


cross_files_d = []
cross_files_l = []

for row in cross_dir:
    for root_d, dirs_d, files_d in os.walk(row):
        if 'csv' in files_d[0]:
            cross_files_l.append(root_d+"/"+files_d[0])
            cross_files_d.append(root_d+"/"+files_d[1])
        else:
            cross_files_d.append(root_d+"/"+files_d[0])
            cross_files_l.append(root_d+"/"+files_d[1])



testing_files_d = []
testing_files_l = []

for row in testing_dir:
    for root_d, dirs_d, files_d in os.walk(row):
        if 'csv' in files_d[0]:
            testing_files_l.append(root_d+"/"+files_d[0])
            testing_files_d.append(root_d+"/"+files_d[1])
        else:
            testing_files_d.append(root_d+"/"+files_d[0])
            testing_files_l.append(root_d+"/"+files_d[1])


#####################################


chunk_time=0.1
down_sample=True
down_sample_rate=4

#overload Audio_data.get_data() and Label.get_labels() to make sure same chunk size and downsampling rate to all data
def get_data(filename):
    return Audio_data.get_data(filename,chunk_time,down_sample,down_sample_rate)
def get_label(filename):
    return Labels.get_labels(filename,chunk_time=chunk_time)

#get the data matrix
(data_1,data_2) = get_data("./Convo_Sample.wav")
test_data=np.concatenate((data_1, data_2), axis=0)

#check dimension
print(test_data.shape)

#reset graph
def reset_graph(seed=1):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

#shuffle data along with label
def shuffle_data(data,label):
    #number of samples
    num=data.shape[0]
    seq=np.random.permutation(num)
    return data[seq],label[seq]

n_inputs = test_data.shape[1]
n_steps=n_inputs//50
n_inputs=n_inputs//n_steps
n_data=n_steps*n_inputs


n_layers = 3
n_neurons = 150
n_outputs = 1
n_epochs = 500
batch_size = 1024
rnn_dropout = 0.5#dropout rate
best_acc = 0

#start seting network
reset_graph()
y = tf.placeholder(tf.float32, shape=(None, n_outputs))
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
training = tf.placeholder_with_default(1.0, shape=[], name='training')
keep_rate=1-rnn_dropout# this is the value used to feed into training, 1 menas no dropout

#recurrent network
lstm_cells = [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons),output_keep_prob=training)
              for layer in range(n_layers)]
multi_cell = tf.contrib.rnn.MultiRNNCell(lstm_cells)
outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)
top_layer_h_state = states[-1][1]

logits = tf.contrib.layers.fully_connected(top_layer_h_state, n_outputs, activation_fn=None)
xentropy=tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = y)
loss = tf.reduce_mean(xentropy)
optimizer = tf.train.AdamOptimizer()
training_op = optimizer.minimize(loss)
pred=tf.cast(logits>0,"int32")


# Test model
correct_prediction = tf.equal(tf.cast(logits>0,"float32"), y)
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


init = tf.global_variables_initializer()
saver = tf.train.Saver()


with tf.Session() as sess:
    init.run()

    for epoch in range(n_epochs):
        training_acc = {}
        cross_acc = {}
        testing_acc = {}

        # Train
        for d, l in zip(training_files_d, training_files_l):
            (data_1, data_2) = get_data(d)
            datas = np.concatenate((data_1, data_2), axis=0)
            datas=datas[:,:n_data]
            datas=datas.reshape(-1,n_steps,n_inputs)
            (label_1, label_2) = get_label(l)
            labels = np.concatenate((label_1, label_2), axis=0)
            training_acc[d]=(accuracy.eval({X: datas, y: labels}))
            epoch_data,epoch_label=shuffle_data(datas,labels)
            for i in range(datas.shape[0]// batch_size):
                X_batch = epoch_data[i:i+batch_size]
                y_batch = epoch_label[i:i+batch_size]
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch,training :keep_rate})
        avg_training_acc=sum(training_acc.values())/len(training_acc)

        # Testing
        for d, l in zip(testing_files_d, testing_files_l):
            (data_1, data_2) = get_data(d)
            Test_datas = np.concatenate((data_1, data_2), axis=0)
            Test_datas = Test_datas[:, :n_data]
            Test_datas = Test_datas.reshape(-1, n_steps, n_inputs)
            (label_1, label_2) = get_label(l)
            Test_labels = np.concatenate((label_1, label_2), axis=0)
            testing_acc[d] = (accuracy.eval({X: Test_datas, y: Test_labels}))
        avg_acc_testing = sum(testing_acc.values()) / len(testing_acc)


        # Cross-Validation
        for d, l in zip(cross_files_d, cross_files_l):
            (data_1, data_2) = get_data(d)
            cross_datas = np.concatenate((data_1, data_2), axis=0)
            cross_datas = cross_datas[:,:n_data]
            cross_datas = cross_datas.reshape(-1, n_steps, n_inputs)
            (label_1, label_2) = get_label(l)
            cross_labels = np.concatenate((label_1, label_2), axis=0)
            cross_acc[d] = (accuracy.eval({X: cross_datas, y: cross_labels}))
        avg_acc_cross = sum(cross_acc.values())/len(cross_acc)

        print(epoch, " Training average is :", avg_training_acc, "Cross - average is :",avg_acc_cross , "Testing average is :", avg_acc_testing)

        saver.save(sess, save_path_latest)
