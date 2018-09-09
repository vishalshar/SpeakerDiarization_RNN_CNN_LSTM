import Audio_data
import Labels
import numpy as np
import tensorflow as tf
from tensorflow.contrib.data import Dataset, Iterator


directory_csv = "/media/vishal/Share/Diarization/Project Data/testData/csv/"
directory_audio = "/media/vishal/Share/Diarization/Project Data/testData/Sound/"

#get the data matrix
(data_1,data_2) = Audio_data.get_data(directory_audio)
datas=np.concatenate((data_1, data_2), axis=0)

#get labels
(label_1,label_2) = Labels.get_labels(directory_csv)
labels=np.concatenate((label_1,label_2),axis=0)

#check dimension
print(labels.shape,datas.shape)


def reset_graph(seed=1):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
    # tf.create_graph()


n_inputs = datas.shape[1]
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 1
n_epochs = 20
batch_size = 2

#start seting network
reset_graph()

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.float32, shape=(None, n_outputs), name="y")
hiddden_1 = tf.contrib.layers.fully_connected(X,n_hidden1)
hiddden_2 = tf.contrib.layers.fully_connected(hiddden_1,n_hidden2)
logits = tf.contrib.layers.fully_connected(hiddden_1, n_outputs, activation_fn=None)
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


# Configure for GPU
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True

# create TensorFlow Dataset objects
# tr_data = Dataset.from_tensor_slices((datas, labels))
# val_data = Dataset.from_tensor_slices((datas, labels))
#
# # create TensorFlow Iterator object
# iterator = Iterator.from_structure(tr_data.output_types,
#                                    tr_data.output_shapes)
# next_element = iterator.get_next()
#
# # create two initialization ops to switch between the datasets
# training_init_op = iterator.make_initializer(tr_data)
# validation_init_op = iterator.make_initializer(val_data)

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    init.run()


    for epoch in range(n_epochs):
        for i in range(datas.shape[0]// batch_size):
            X_batch = datas[i:i+batch_size]
            y_batch = labels[i:i+batch_size]
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            #print(i, "loss:", loss.eval(feed_dict={X: X_batch, y: y_batch}))
        print("Accuracy:", accuracy.eval({X: datas, y: labels}))
    out = pred.eval({X: datas})[100:200]
    # print(np.concatenate((out,labels[100:200]),1))


