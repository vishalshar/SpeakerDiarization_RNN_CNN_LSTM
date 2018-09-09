# 3 main functions in this code, uncommon each function to use
#train_and_save_network() train the network and save resuelt with early stopping
#test_with_restore_data() test results on test data set
#compare(data_file,label_file) com predicted result with true labels

import Load_Audio_Data
import Labels
import numpy as np
import tensorflow as tf
import os
import random
import sys
import matplotlib.pyplot as plt
import datetime

#settings, pathes
np.random.seed(2)
chunk_time=0.1
down_sample=True
down_sample_rate=4
rate=44100
fs=rate/down_sample_rate
dir_raw_data="./data/Sound_Files/"
dir_normalized_data="./data/Normalized_Sound_Files/"
dir_raw_data=dir_normalized_data#Use normalized data instead of raw data
dir_label="./data/Cleaned_Labels/"

#save pathes
save_path="./Alg4_1channel/ALg4_best_acc.ckpt"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
save_path_latest="./Alg4_1channel/ALg4_latest_acc.ckpt"
os.makedirs(os.path.dirname(save_path_latest), exist_ok=True)


#overload Audio_data.get_data() and Label.get_labels() to make sure same chunk size and downsampling rate to all data
def get_data(filename):
    d1,d2=Load_Audio_Data.get_data(filename,chunk_time,down_sample,down_sample_rate)
    return np.concatenate((d1,d2),axis=0)
def get_label(filename):
    l1,l2=Labels.get_labels(filename, chunk_time=chunk_time)
    return np.concatenate((l1,l2),axis=0)

#get the data matrix
example_data = get_data("./Convo_Sample.wav")

#check dimension
print(example_data.shape)

def divide_data(dir_data,dir_label):
    for (root_d, dirs_d, files_d), (root_l, dirs_l, files_l) in zip(os.walk(dir_data), os.walk(dir_label)):
        full_files_d=[root_d+x for x in files_d]
        full_files_l = [root_l + x for x in files_l]
        pairs=list(zip(full_files_d,full_files_l))
        np.random.shuffle(pairs)
        num=len(pairs)
        train=pairs[:int(num*0.7)]
        cross=pairs[int(num*0.7):int(num*0.85)]
        test=pairs[int(num*0.85):]
    return train,cross,test

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

train_set,cross_set,test_set=divide_data(dir_raw_data,dir_label)

n_data = example_data.shape[1]
n_steps=40
n_inputs=n_data//n_steps
n_data=n_steps*n_inputs

n_layers = 2
n_neurons = 100
n_outputs = 1
n_epochs = 1000
batch_size = 256
rnn_dropout=0#dropout rate, 0 means no dropout

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


#test_with_restore_data()
def train_and_save_network():
    best_acc = 0
    with tf.Session() as sess:
        init.run()

        for epoch in range(n_epochs):
            #train on all training data and get training accuracy
            np.random.shuffle(train_set)
            print("training on ", [d.split("/")[-1] for d, l in train_set[:5]])
            whole_data = np.concatenate([get_data(x[0]) for x in train_set[:5]], axis=0)
            whole_data = whole_data[:, :n_data]
            whole_data = whole_data.reshape(-1, n_steps, n_inputs)
            whole_label = np.concatenate([get_label(x[1]) for x in train_set[:5]], axis=0)
            training_acc = {}
            cross_acc = {}
            epoch_data, epoch_label = shuffle_data(whole_data, whole_label)

            for i in range(epoch_data.shape[0] // batch_size):
                X_batch = epoch_data[i:i + batch_size]
                y_batch = epoch_label[i:i + batch_size]
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch, training: keep_rate})
                if i%100==0:print(i," batches trained")

            print("loss:", loss.eval(feed_dict={X: X_batch, y: y_batch}))
            for d,l in train_set[:5]:
                data=get_data(d)[:, :n_data]
                data=data.reshape(-1, n_steps, n_inputs)
                training_acc[d.split("/")[-1]]=accuracy.eval({X: data, y: get_label(l)})
            avg_training_acc = sum(training_acc.values()) / len(training_acc)
            print("epoch:", epoch, "training Accuracy:", training_acc, "\n average is :", avg_training_acc)


            #testing
            for d, l in cross_set:
                cross_data = get_data(d)
                cross_data=cross_data[:, :n_data]
                cross_data = cross_data.reshape(-1, n_steps, n_inputs)
                cross_label = get_label(l)
                cross_acc[d.split("/")[-1]] = (accuracy.eval({X: cross_data, y: cross_label}))
            avg_acc = sum(cross_acc.values()) / len(cross_acc)
            print("epoch:", epoch, "Cross Accuracy:", cross_acc, "\n average is :", avg_acc)
            if avg_acc > best_acc:
                best_acc = avg_acc
                saver.save(sess, save_path)
                print("Network saved")
            saver.save(sess, save_path_latest)
        print("max training acc is", best_acc)

train_and_save_network()


