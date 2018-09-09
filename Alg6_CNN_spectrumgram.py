#It may be faster if pre-compute the spectrumgram and store it

import Audio_data
import Labels
import numpy as np
from scipy import signal
import tensorflow as tf
import os
import random
import sys
import matplotlib.pyplot as plt
import datetime
import pickle

#dir of data and labels
#dir for training
dir_data="./data/Datas/"
dir_spectrum_data="./data/Spectrum_Datas/"
os.makedirs(os.path.dirname(dir_spectrum_data), exist_ok=True)
dir_label="./data/Labels/"

#dir for cross
dir_cross_data="./data/Cross_datas/"
dir_spectrum_cross_data="./data/Spectrum_Cross_datas/"
os.makedirs(os.path.dirname(dir_spectrum_cross_data), exist_ok=True)
dir_cross_label="./data/Cross_labels/"

#dir for test
dir_test_data="./data/Test_datas/"
dir_test_label="./data/Test_labels/"
dir_spectrum_test_data="./data/Spectrum_Test_datas/"
os.makedirs(os.path.dirname(dir_spectrum_test_data), exist_ok=True)

#save pathes
save_path="./Alg6/ALg6_best_acc.ckpt"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
save_path_latest="./Alg6/ALg6_latest_acc.ckpt"
os.makedirs(os.path.dirname(save_path_latest), exist_ok=True)

chunk_time=0.1
down_sample=True
down_sample_rate=4
rate=44100
fs=rate/down_sample_rate

#overload Audio_data.get_data() and Label.get_labels() to make sure same chunk size and downsampling rate to all data
def get_data(filename):
    return Audio_data.get_data(filename,chunk_time,down_sample,down_sample_rate)
def get_label(filename):
    return Labels.get_labels(filename,chunk_time=chunk_time)

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

#compute spectrumgram
def compute_spectrungram(data,fs):
    sg = []
    for i in range(data.shape[0]):
        f, t, Sxx = signal.spectrogram(data[i], fs)
        sg.append(Sxx)
    return np.array(sg)

#precompute spectrumgram for the training data set
def pre_process_data(fs,dir_source,dir_out):
    for (root_d, dirs_d, files_d) in os.walk(dir_source):
        for d in files_d:
            (data_1, data_2) = get_data(root_d + d)
            datas = np.concatenate((data_1, data_2), axis=0)
            spectrum_gram = compute_spectrungram(datas, fs)
            a,b=d.split(".")
            pickle.dump(spectrum_gram, open(dir_out+a+".p", "wb"))

data_1,data_2 = get_data("./Convo_Sample.wav")

sg=compute_spectrungram(data_1,fs) #in shape(none,height,width)
print("compute_spectrungram shape is", sg.shape)

height = sg.shape[1]
width = sg.shape[2]
print("hight is ",height,"width is ",width)
n_inputs=height*width
channels=1

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
    logits = tf.contrib.layers.fully_connected(fc1, n_outputs, activation_fn=None)

with tf.name_scope("train"):
    xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct_prediction = tf.equal(tf.cast(logits > 0, "float32"), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    pred = tf.cast(logits > 0, "int32")

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

n_epochs = 1000
batch_size = 256


#check dir before use
#after trining, use this function to test
def test_with_restore_data():
    with tf.Session() as sess:
        saver.restore(sess, save_path_latest) # or better, use save_path
        for (root_d, dirs_d, files_d), (root_l, dirs_l, files_l) in zip(os.walk(dir_test_data), os.walk(dir_test_label)):
            testing_acc = {}
            for d, l in zip(files_d, files_l):
                (data_1, data_2) = get_data(root_d + d)
                Test_datas = np.concatenate((data_1, data_2), axis=0)
                spectrum_gram = compute_spectrungram(Test_datas, fs)
                (label_1, label_2) = get_label(root_l + l)
                Test_labels = np.concatenate((label_1, label_2), axis=0)
                testing_acc[d] = (accuracy.eval({X: spectrum_gram, y: Test_labels}))
        avg_acc = sum(testing_acc.values()) / len(testing_acc)
        print( "testing Accuracy:", testing_acc, "\n average is :", avg_acc)
    print("Testing end")


#check dir before use
# after training, use this function observe predict and result
def compare(data_file,label_file):
    with tf.Session() as sess:
        saver.restore(sess, save_path) # or better, use save_path
        (data_1, data_2) = get_data(data_file)
        Test_datas = np.concatenate((data_1, data_2), axis=0)
        spectrum_gram = compute_spectrungram(Test_datas, fs)
        (label_1, label_2) = get_label(label_file)
        Test_labels = np.concatenate((label_1, label_2), axis=0)
        out = pred.eval({X: spectrum_gram})
    print(Test_labels[0:300])

    #starting time point in unit of chunck size
    start_point=0
    for x,y,z in zip((421,423,425,427),(422,424,426,428),(start_point,start_point+300,start_point+600,start_point+900)):
        plt.subplot(x)
        plt.title("Label")
        j=-1
        for i in Test_labels[z:z+300]:
            j+=1
            if i:
                plt.axvline(j/10)
        plt.xlim((0,31))

        plt.subplot(y)
        plt.title("Prediction")
        j=-1
        for i in out[z:z+300]:
            j+=1
            if i:
                plt.axvline(j/10)
        plt.xlim((0,31))
    plt.show()

#compare("./data/Test_datas/HS_D30.wav","./data/Test_labels/HS_D30.csv")

def train_and_save_network():
    best_acc = 0
    pre_process_data(fs, dir_data, dir_spectrum_data) # pre-compute training set
    pre_process_data(fs, dir_cross_data, dir_spectrum_cross_data)  # pre-compute cross set
    pre_process_data(fs, dir_test_data, dir_spectrum_test_data)  # pre-compute test set
    with tf.Session() as sess:
        init.run()

        for epoch in range(n_epochs):
            #train on training set
            for (root_d, dirs_d, files_d), (root_l, dirs_l, files_l) in zip(os.walk(dir_spectrum_data), os.walk(dir_label)):
                training_acc = {}
                cross_acc = {}
                pairs = list(zip(files_d, files_l))
                random.shuffle(pairs)
                for d, l in pairs:
                    datas = pickle.load( open(root_d+d, "rb" ) )
                    (label_1, label_2) = get_label(root_l + l)
                    labels = np.concatenate((label_1, label_2), axis=0)
                    epoch_data, epoch_label = shuffle_data(datas, labels)
                    for i in range(datas.shape[0] // batch_size):
                        X_batch = epoch_data[i:i + batch_size]
                        y_batch = epoch_label[i:i + batch_size]
                        sess.run(training_op, feed_dict={X: X_batch, y: y_batch, training: False})
                    training_acc[d] = (accuracy.eval({X: datas, y: labels}))
            print("loss:", loss.eval(feed_dict={X: X_batch, y: y_batch}))
            avg_training_acc = sum(training_acc.values()) / len(training_acc)
            print("epoch:", epoch, "training Accuracy:", training_acc, "\n average is :", avg_training_acc)

            #check cross accuracy
            for (root_d, dirs_d, files_d), (root_l, dirs_l, files_l) in zip(os.walk(dir_spectrum_cross_data),
                                                                            os.walk(dir_cross_label)):
                for d, l in zip(files_d, files_l):
                    cross_datas = pickle.load(open(root_d+d, "rb"))
                    (label_1, label_2) = get_label(root_l + l)
                    cross_labels = np.concatenate((label_1, label_2), axis=0)
                    cross_acc[d] = (accuracy.eval({X: cross_datas, y: cross_labels}))
            avg_acc = sum(cross_acc.values()) / len(cross_acc)
            print("epoch:", epoch, "Cross Accuracy:", cross_acc, "\n average is :", avg_acc)
            if avg_acc > best_acc:
                best_acc = avg_acc
                saver.save(sess, save_path)
                print("Network saved")
            saver.save(sess, save_path_latest)
            test_with_restore_data()
        print("max training acc is", best_acc)


train_and_save_network()
#test_with_restore_data()