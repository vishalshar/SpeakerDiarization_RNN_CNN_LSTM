import Audio_data
import Labels
import numpy as np
import tensorflow as tf
import os
import random
import sys
import matplotlib.pyplot as plt
import datetime

#benchmark accuracy is %62 (always guess 0)

#dir of data and labels
dir_data="./data/Datas/"
dir_label="./data/Labels/"
dir_cross_data="./data/Cross_datas/"
dir_cross_label="./data/Cross_labels/"
dir_test_data="./data/Test_datas/"
dir_test_label="./data/Test_labels/"

chunk_time=0.1
down_sample=True
down_sample_rate=4
early_stoping=False

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
n_hidden1 = 200
n_hidden2 = 50
n_hidden3=20
n_hidden4=20
n_outputs = 1
n_epochs = 1000
batch_size = 256
h1_dropout_rate=0.0
h2_dropout_rate=0.0
h3_dropout_rate=0.0
best_acc=0

#start seting network
reset_graph()

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.float32, shape=(None, n_outputs), name="y")
training = tf.placeholder_with_default(False, shape=[], name='training')

hiddden_1 = tf.contrib.layers.fully_connected(X,n_hidden1)
drop_hidden_1 = tf.layers.dropout(hiddden_1, h1_dropout_rate,training=training)
hiddden_2 = tf.contrib.layers.fully_connected(hiddden_1,n_hidden2)
drop_hidden_2 = tf.layers.dropout(hiddden_2, h2_dropout_rate,training=training)
#hiddden_3 = tf.contrib.layers.fully_connected(hiddden_2,n_hidden3)
#drop_hidden_3 = tf.layers.dropout(hiddden_3, h3_dropout_rate,training=training)
#hiddden_4 = tf.contrib.layers.fully_connected(hiddden_3,n_hidden4)
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

#observe predict and result
def compare(data_file,label_file):
    with tf.Session() as sess:
        saver.restore(sess, "./Alg1/Alg1_best_acc.ckpt") # or better, use save_path
        (data_1, data_2) = get_data(data_file)
        Test_datas = np.concatenate((data_1, data_2), axis=0)
        (label_1, label_2) = get_label(label_file)
        Test_labels = np.concatenate((label_1, label_2), axis=0)
        out = pred.eval({X: Test_datas})
    print(Test_labels[0:300])

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

def test_with_restore_data():
    with tf.Session() as sess:
        saver.restore(sess, "./Alg1/Alg1_best_acc.ckpt") # or better, use save_path
        for (root_d, dirs_d, files_d), (root_l, dirs_l, files_l) in zip(os.walk(dir_test_data), os.walk(dir_test_label)):
            testing_acc = {}
            for d, l in zip(files_d, files_l):
                (data_1, data_2) = get_data(root_d + d)
                Test_datas = np.concatenate((data_1, data_2), axis=0)
                (label_1, label_2) = get_label(root_l + l)
                Test_labels = np.concatenate((label_1, label_2), axis=0)
                testing_acc[d] = (accuracy.eval({X: Test_datas, y: Test_labels}))
        avg_acc = sum(testing_acc.values()) / len(testing_acc)
        print( "testing Accuracy:", testing_acc, "\n average is :", avg_acc)
        out = pred.eval({X: Test_datas})

    print("Testing end")
    sys.exit()

#test_with_restore_data()

with tf.Session() as sess:
    init.run()

    for epoch in range(n_epochs):
        #train on all training data and get training accuracy
        for (root_d, dirs_d, files_d), (root_l, dirs_l, files_l) in zip(os.walk(dir_data), os.walk(dir_label)):
            training_acc = {}
            cross_acc = {}
            pairs=list(zip(files_d,files_l))
            random.shuffle(pairs)

            for d, l in pairs:
                (data_1, data_2) = get_data(root_d+d)
                datas = np.concatenate((data_1, data_2), axis=0)
                (label_1, label_2) = get_label(root_l+l)
                labels = np.concatenate((label_1, label_2), axis=0)
                training_acc[d]=(accuracy.eval({X: datas, y: labels, training: True}))
                epoch_data,epoch_label=shuffle_data(datas,labels)
                for i in range(datas.shape[0]// batch_size):
                    X_batch = epoch_data[i:i+batch_size]
                    y_batch = epoch_label[i:i+batch_size]
                    sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        print(i, "loss:", loss.eval(feed_dict={X: X_batch, y: y_batch}))
        print("epoch:", epoch, "training Accuracy:", training_acc,"\n average is :",sum(training_acc.values())/len(training_acc))

        #testing
        for (root_d, dirs_d, files_d), (root_l, dirs_l, files_l) in zip(os.walk(dir_cross_data), os.walk(dir_cross_label)):
            for d, l in zip(files_d, files_l):
                (data_1, data_2) = get_data(root_d + d)
                Test_datas = np.concatenate((data_1, data_2), axis=0)
                (label_1, label_2) = get_label(root_l + l)
                Test_labels = np.concatenate((label_1, label_2), axis=0)
                cross_acc[d] = (accuracy.eval({X: Test_datas, y: Test_labels}))
        avg_acc=sum(cross_acc.values())/len(cross_acc)
        print("epoch:",epoch,"Cross Accuracy:", cross_acc,"\n average is :",avg_acc)
        if early_stoping and avg_acc>best_acc:
            best_acc=avg_acc
            save_path = saver.save(sess, "./Alg1/Alg1_best_acc.ckpt")
    #out=pred.eval({X: datas})[100:200]
    #print(np.concatenate((out,labels[100:200]),1))



