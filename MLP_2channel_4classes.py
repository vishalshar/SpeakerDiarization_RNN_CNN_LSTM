import Audio_data
import Labels
import numpy as np
import tensorflow as tf
import os
import random



#dir of data and labels
dir_data="./data/Datas/"
dir_label="./data/Labels/"
dir_test_data="./data/Test_datas/"
dir_test_label="./data/Test_labels/"

chunk_time=0.1
down_sample=True
down_sample_rate=4

#overload Audio_data.get_data() to make sure same chunk size and downsampling rate to all data
def get_data(filename):
    data_1,data_2=Audio_data.get_data(filename,chunk_time,down_sample,down_sample_rate)
    return np.concatenate((data_1,data_2),axis=1)

#overload Labels.get_labels for dimension problem, and expand the labels to 4 classes
#0: no speech 1:intended speaker 2:another speaker 3: overlap of two speaker
def get_label(filename):
    temp1,temp2=Labels.get_labels(filename,chunk_time=chunk_time)
    temp1, temp2 =temp1.reshape(-1), temp2.reshape(-1)
    return temp1+temp2*2

#get a sample data matrix
data = get_data("./Convo_Sample.wav")

#check dimension
print(data.shape)

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


n_inputs = data.shape[1]
n_hidden1 = 200
n_hidden2 = 50
n_outputs = 4
n_epochs = 1000
batch_size = 128
h1_dropout_rate=1
h2_dropout_rate=1

#start seting network
reset_graph()

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")
hiddden_1 = tf.contrib.layers.fully_connected(X,n_hidden1)
drop_hidden_1 = tf.layers.dropout(hiddden_1, h1_dropout_rate)
#uncomment this block if need second hidden layer
#hiddden_2 = tf.contrib.layers.fully_connected(hiddden_1,n_hidden2)
#drop_hidden_2 = tf.layers.dropout(hiddden_2, h2_dropout_rate)
logits = tf.contrib.layers.fully_connected(hiddden_1, n_outputs, activation_fn=None)
xentropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = y)
loss = tf.reduce_mean(xentropy)
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
training_op = optimizer.minimize(loss)
pred=tf.arg_max(logits,1)


# Test model
correct_prediction = tf.nn.in_top_k(logits, y, 1)

# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

#initialized network and prepare saver
init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    init.run()

    #this loop go through each epoch, in each epoch, each training data is trained once
    for epoch in range(n_epochs):
        #train on all training data and get training accuracy
        for (root_d, dirs_d, files_d), (root_l, dirs_l, files_l) in zip(os.walk(dir_data), os.walk(dir_label)):
            training_acc = {}
            testing_acc = {}
            pairs=list(zip(files_d,files_l))
            random.shuffle(pairs)

            for d, l in pairs:
                data=get_data(root_d+d)
                label=get_label(root_l+l)
                epoch_data, epoch_label = shuffle_data(data, label)
                for i in range(epoch_data.shape[0]// batch_size):
                    X_batch = epoch_data[i:i+batch_size]
                    y_batch = epoch_label[i:i+batch_size]
                    sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
                training_acc[d] = (accuracy.eval({X: data, y: label}))
        print( "loss:", loss.eval(feed_dict={X: data, y: label}))
        print("epoch:", epoch, "training Accuracy:", training_acc,"average is :",sum(training_acc.values())/len(training_acc))

        #testing
        for (root_d, dirs_d, files_d), (root_l, dirs_l, files_l) in zip(os.walk(dir_test_data), os.walk(dir_test_label)):
            for d, l in zip(files_d, files_l):
                test_data=get_data(root_d + d)
                test_label=get_label(root_l + l)
                testing_acc[d] = (accuracy.eval({X: test_data, y: test_label}))
        print("testing: epoch:", epoch, "testing Accuracy:", testing_acc, "average is :",
              sum(testing_acc.values()) / len(testing_acc))
    save_path = saver.save(sess, "./4classes/my_model_final.ckpt")
    #out=pred.eval({X: datas})[100:200]
    #print(np.concatenate((out,labels[100:200]),1))


