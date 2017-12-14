#just put all data and labes in below, then run this file
#dir_raw_data="./data/Sound_Files/"
#dir_label="./data/Cleaned_Labels/"

import Spectrogram_Generator
import Labels
import numpy as np
import os
import pickle
import tensorflow as tf

#settings, pathes
np.random.seed(2)
chunk_time=0.1
down_sample=True
down_sample_rate=4
rate=44100
fs=rate/down_sample_rate
dir_raw_data="./data/Datas/"
dir_normalized_data="./data/Normalized_Sound_Files/"
dir_raw_data=dir_normalized_data#Use normalized data instead of raw data
dir_spec_data="./data/Spectrograms_1channel-fs-"+str(int(fs))+"/"
dir_label="./data/Cleaned_Labels/"

#save pathes
save_path="./Alg6_1channel/ALg6_best_acc.ckpt"
# os.makedirs(os.path.dirname(save_path))
save_path_latest="./Alg6_1channel/ALg6_latest_acc.ckpt"
# os.makedirs(os.path.dirname(save_path_latest))


#load spectrogram
#the demision of returned matrix is (num_of_chuncks,height, width,channels)
def get_data(filename):
    return pickle.load( open(filename, "rb" ) )
#overload Labels.get_labels for dimension problem, and expand the labels to 4 classes
#0: no speech 1: speaker_1(speaker of channel 1) 2:speaker_2 3: overlap of two speaker
def get_label(filename):
    l1,l2=Labels.get_labels(filename, chunk_time=chunk_time)
    return np.concatenate((l1,l2),axis=0)

#divide whole data set into train, cross and test
#return 3 lists of file path for each group, each element in a list is (data, label)
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

# to shuffle data in each eapoch
def shuffle_data(data,label):
    #number of samples
    num=data.shape[0]
    seq=np.random.permutation(num)
    return data[seq],label[seq]

# to reset graph
def reset_graph(seed=1):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


#generate spectrogram if not exist
if not os.path.isdir(dir_spec_data):
    Spectrogram_Generator.convert_to_spectrogram(dir_raw_data,dir_spec_data,
                        fs,output_channel=1,chunk_time=chunk_time, down_sample=down_sample,down_sample_rate=down_sample_rate)
#make train, cross, test sets, each of them is a list of (data,label)
train_set,cross_set,test_set=divide_data(dir_spec_data,dir_label)

#pick one data file to get demisions
example_data=get_data(train_set[0][0])
print(example_data.shape,get_label(train_set[0][1]).shape)
height=example_data.shape[1]
width=example_data.shape[2]
channels=1
print("hight is ",height,"width is ",width)

#set CNN network
conv1_fmaps = 32
conv1_ksize = 2
conv1_stride = 1
conv1_pad = "SAME"

conv2_fmaps = 64
conv2_ksize = 2
conv2_stride = 1
conv2_pad = "SAME"
conv2_dropout_rate = 0

pool3_fmaps = conv2_fmaps

n_fc1 = 128
fc1_dropout_rate = 0 # dropout rate  to tune
n_outputs = 1

reset_graph()

with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, shape=[None, height, width], name="X")
    X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
    y = tf.placeholder(tf.float32, shape=(None))
    training = tf.placeholder_with_default(False, shape=[], name='training')

conv1 = tf.layers.conv2d(X_reshaped, filters=conv1_fmaps, kernel_size=conv1_ksize,
                         strides=conv1_stride, padding=conv1_pad,
                         activation=tf.nn.relu, name="conv1")
conv2 = tf.layers.conv2d(conv1, filters=conv2_fmaps, kernel_size=conv2_ksize,
                         strides=conv2_stride, padding=conv2_pad,
                         activation=tf.nn.relu, name="conv2")

with tf.name_scope("pool3"):
    pool3 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    pool3_flat = tf.reshape(pool3, shape=[-1, pool3_fmaps * (height//2) * (width//2)])
    pool3_flat_drop = tf.layers.dropout(pool3_flat, conv2_dropout_rate, training=training)

with tf.name_scope("fc1"):
    fc1 = tf.layers.dense(pool3_flat_drop, n_fc1, activation=tf.nn.relu, name="fc1")
    fc1_drop = tf.layers.dropout(fc1, fc1_dropout_rate, training=training)

with tf.name_scope("output"):
    logits = tf.contrib.layers.fully_connected(fc1_drop, n_outputs, activation_fn=None)

with tf.name_scope("train"):
    xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct_prediction =tf.equal(tf.cast(logits > 0, "float32"), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    pred = tf.cast(logits > 0, "int32")

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

n_epochs = 1000
batch_size = 256

def train_and_save_network():
    best_acc = 0
    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            #train on training set
            #in each epoch, randomly read 5 files
            np.random.shuffle(train_set)
            print("training on ",[d.split("/")[-1] for d,l in train_set[:5]])
            whole_data = np.concatenate([get_data(x[0]) for x in train_set[:5]], axis=0)
            whole_label = np.concatenate([get_label(x[1]) for x in train_set[:5]], axis=0)
            training_acc = {}
            cross_acc = {}
            epoch_data,epoch_label = shuffle_data(whole_data,whole_label)
            #epoch_data, epoch_label=whole_data,whole_label
            for i in range(epoch_data.shape[0] // batch_size):
                X_batch = epoch_data[i:i + batch_size]
                y_batch = epoch_label[i:i + batch_size]
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch, training: True})
                if i%100==0:print(i," batches trained")

            print("loss:", loss.eval(feed_dict={X: X_batch, y: y_batch}))
            for d,l in train_set[:5]:
                training_acc[d.split("/")[-1]]=accuracy.eval({X: get_data(d), y: get_label(l)})
            avg_training_acc = sum(training_acc.values()) / len(training_acc)
            print("epoch:", epoch, "training Accuracy:", training_acc, "\n average is :", avg_training_acc)

            #check cross accuracy
            for d,l in cross_set:
                cross_data = get_data(d)
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
