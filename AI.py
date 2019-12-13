#Adapted from NRK96 EMNIST Classifier
#To work with handwritten letters on python 3.7.3 and tensorflow 1.14.0
#By Tanner M and Tyler M 

#imports the input data function from tensorflow
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#loads emnist from a directory(edit to match where emnist gzip file is)

emnist = input_data.read_data_sets("D:\Software\Scripts\gzip", one_hot = True)

sess = tf.InteractiveSession()

#outputs set to 26 since that is the EMNIST dataset for just letters
#MUST CHANGE tensorflow/contrib/learn/python/learn/datasets/mnist.py extract_labels function
#   to have the same number of output classes
#output results to text file
outputs = 26
text = "Results.txt"

#weight variable
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

#bias variable
def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

#convolution layer
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

#pooling layer
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1],
                          strides = [1, 2, 2, 1], padding = 'SAME')

#set initial x value
x = tf.placeholder(tf.float32, shape = [None, 784])
#set expected outputs
y_ = tf.placeholder(tf.float32, shape = [None, outputs])
#creates convolution layer from inputs
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, 28, 28, 1])
#hooks input to hidden layer 1 and applies a relu function
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
#hooks hidden layer 1 to hidden layer 2, and applies a relu function
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
#hooks hidden layer 2 to the final hidden layer and applies a relu function
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32)
#applies dropout
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
W_fc2 = weight_variable([1024, outputs])
b_fc2 = bias_variable([outputs])
#output
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#applies cross entropy loss on the expected and actual values
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y_conv))
#applies the Adam Optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#works as main function to run tensorflow
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #batches data and trains
    print('\n Training model ... ')
    for i in range(20000):
        batch = emnist.train.next_batch(50)
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict = {
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            #evaluate feedback every 100 epochs
            print('step %d, training accuracy %g' % (i, train_accuracy))
            print('step %d, training accuracy %g' % (i, train_accuracy), file = open(text, "a"))
        train_step.run(feed_dict = {x: batch[0], y_: batch[1], keep_prob: 0.6})
    print('\n\n Testing model ...')
    tot_accuracy = 0
    #batches data and tests
    for i in range(20000):
        batch = emnist.test.next_batch(50)
        if i%100 == 0:
            test_accuracy = accuracy.eval(feed_dict = {
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            #evaluate feedback every 100 epochs
            print('step %d, test accuracy %g' % (i, test_accuracy))
            print('step %d, test accuracy %g' % (i, test_accuracy), file = open(text, "a"))
            tot_accuracy = tot_accuracy + test_accuracy
    #find total accuracy and report it
    final = tot_accuracy/200.0
    print('Final test accuracy: %f' % final)
    print('Final test accuracy: %f' % final, file = open(text, "a"))