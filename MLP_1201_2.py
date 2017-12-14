import os

import numpy as np
import tensorflow as tf

import Audio_data
from NN_Convo_1201_2 import Labels
import random

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



#overload Audio_data.get_data() and Label.get_labels() to make sure same chunk size and downsampling rate to all data
def get_data(filename):
    return Audio_data.get_data(filename,chunk_time,down_sample,down_sample_rate)
def get_label(filename):
    return Labels.get_labels(filename,chunk_time=chunk_time)


chunk_time=0.1
down_sample=True
down_sample_rate=4
#overload Audio_data.get_data() to make sure same chunk size and downsampling rate to all data
def get_data(filename):
    return Audio_data.get_data(filename,chunk_time,down_sample,down_sample_rate)


#shuffle data along with label
def shuffle_data(data,label):
    #number of samples
    num=data.shape[0]
    seq=np.random.permutation(num)
    return data[seq],label[seq]

#get the data matrix
(data_1,data_2) = get_data("./Convo_Sample.wav")
test_data=np.concatenate((data_1, data_2), axis=0)



n_inputs = test_data.shape[1]
n_hidden1 = 75
n_hidden2 = 25
n_outputs = 1
n_epochs = 500
batch_size = 1024
h1_dropout_rate = 0.1
h2_dropout_rate = 0.1

print "n_hidden1 "+str(n_hidden1), "n_outputs "+str(n_outputs), "n_epochs "+str(n_epochs), "batch_size "+str(batch_size)

# to reset graph
def reset_graph(seed=1):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


# Current: h1_dropout_rate = 0.5
#start seting network
reset_graph()

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.float32, shape=(None, n_outputs), name="y")
hiddden_1 = tf.contrib.layers.fully_connected(X,n_hidden1)
drop_hidden_1 = tf.layers.dropout(hiddden_1, h1_dropout_rate)
hiddden_2 = tf.contrib.layers.fully_connected(drop_hidden_1,n_hidden2)
drop_hidden_2 = tf.layers.dropout(hiddden_2, h2_dropout_rate)
logits = tf.contrib.layers.fully_connected(drop_hidden_2, n_outputs, activation_fn=None)
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
        # Train
        training_acc = {}
        testing_acc = {}
        cross_acc = {}

        for d, l in zip(training_files_d, training_files_l):
            (data_1, data_2) = get_data(d)
            datas = np.concatenate((data_1, data_2), axis=0)
            (label_1, label_2) = Labels.get_labels(l)
            labels = np.concatenate((label_1, label_2), axis=0)
            training_acc[d]=(accuracy.eval({X: datas, y: labels}))
            epoch_data,epoch_label=shuffle_data(datas,labels)
            for i in range(datas.shape[0]// batch_size):
                X_batch = epoch_data[i:i+batch_size]
                y_batch = epoch_label[i:i+batch_size]
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        training_acc_temp = sum(training_acc.values())/len(training_acc)

        # Testing
        for d, l in zip(testing_files_d, testing_files_l):
            (data_1, data_2) = get_data(d)
            Test_datas = np.concatenate((data_1, data_2), axis=0)
            (label_1, label_2) = Labels.get_labels(l)
            Test_labels = np.concatenate((label_1, label_2), axis=0)
            testing_acc[d] = (accuracy.eval({X: Test_datas, y: Test_labels}))
        testing_acc_temp = sum(testing_acc.values())/len(testing_acc)


        # Cross-Validation
        for d, l in zip(cross_files_d, cross_files_l):
            (data_1, data_2) = get_data(d)
            cross_datas = np.concatenate((data_1, data_2), axis=0)
            (label_1, label_2) = Labels.get_labels(l)
            cross_labels = np.concatenate((label_1, label_2), axis=0)
            cross_acc[d] = (accuracy.eval({X: cross_datas, y: cross_labels}))
        avg_acc=sum(cross_acc.values())/len(cross_acc)

        print(epoch, " Training average is :", training_acc_temp, "Cross - average is :",avg_acc , "Testing average is :", testing_acc_temp)
    # save_path = saver.save(sess, "./my_model_final.ckpt")
    #out=pred.eval({X: datas})[100:200]
    #print(np.concatenate((out,labels[100:200]),1))

