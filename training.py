from __future__ import print_function

import os
import time
import tensorflow as tf

import cnn_kicker_model

# Parameters
learning_rate = 0.001
batch_size = 128
training_iters = 39000
display_step = 100
save_step_num = int(training_iters / 5)
save_step = [save_step_num, save_step_num * 2, save_step_num * 3, save_step_num * 4, training_iters]
model_path = './models/CNN'

# Network Parameters
n_classes = 15  # total classes
dropout = 1.0  # Dropout, probability to keep units
global_step = tf.Variable(0, name='global_step', trainable=False)  # 计数器变量，保存模型用，设置为不需训练
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

# Read data
type_list = ['training', 'test']
base_path = 'D:/data_source/kicker'
file_path = '/'.join((base_path, type_list[0], 'kicker_training_*.tfrecords'))
reader = tf.TFRecordReader()
filename = tf.train.match_filenames_once(file_path)
filename_queue = tf.train.string_input_producer(filename)
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
    serialized_example,
    features={
        'input': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.string),
    })
_input = tf.decode_raw(features['input'], tf.uint8)
_input = tf.cast(tf.reshape(_input, shape=[3, 9, 15]), dtype=tf.float32)
_label = tf.decode_raw(features['label'], tf.uint8)
_label = tf.cast(tf.reshape(_label, shape=[15]), dtype=tf.float32)
x, y = tf.train.shuffle_batch(
    [_input, _label],
    batch_size=batch_size,
    num_threads=4,
    capacity=5000 + 10 * batch_size,
    min_after_dequeue=2000)

# Store layers weight & bias
weights = {
    'wc1': tf.Variable(tf.random_normal([3, 3, 3, 16], stddev=0.05)),
    'wc2': tf.Variable(tf.random_normal([3, 3, 16, 32], stddev=0.05)),
    'wc3': tf.Variable(tf.random_normal([3, 1, 32, 64], stddev=0.05)),
    'wc4': tf.Variable(tf.random_normal([3, 1, 64, 64], stddev=0.05)),
    'wc5': tf.Variable(tf.random_normal([3, 1, 64, 64], stddev=0.05)),
    # fully connected
    'wd1': tf.Variable(tf.random_normal([15 * 64, 512], stddev=0.04)),
    # 512 inputs, 309 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([512, n_classes], stddev=1 / 512.0))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([16])),
    'bc2': tf.Variable(tf.random_normal([32])),
    'bc3': tf.Variable(tf.random_normal([64])),
    'bc4': tf.Variable(tf.random_normal([64])),
    'bc5': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([512])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

restore_var = {
    'wc1': weights['wc1'],
    'wc2': weights['wc2'],
    'wc3': weights['wc3'],
    'wc4': weights['wc4'],
    'wc5': weights['wc5'],
    'wd1': weights['wd1'],
    'wout': weights['out'],
    'bc1': biases['bc1'],
    'bc2': biases['bc2'],
    'bc3': biases['bc3'],
    'bc4': biases['bc4'],
    'bc5': biases['bc5'],
    'bd1': biases['bd1'],
    'bout': biases['out']
}

# Construct model
pred = cnn_kicker_model.conv_net(x, weights, biases, keep_prob)

sc = tf.get_collection("scale")
bt = tf.get_collection("beta")
pm = tf.get_collection("pop_mean")
pv = tf.get_collection("pop_var")
for i in range(len(sc)):
    restore_var['scale' + str(i)] = sc[i]
    restore_var['beta' + str(i)] = bt[i]
    restore_var['pop_mean' + str(i)] = pm[i]
    restore_var['pop_var' + str(i)] = pv[i]

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# Initializing the variables
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

tf.add_to_collection('inputs', x)
tf.add_to_collection('inputs', y)
tf.add_to_collection('pred', pred)

# save models
ckpt_dir = model_path + time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
saver = tf.train.Saver(restore_var)

# 分配显存
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
config.gpu_options.per_process_gpu_memory_fraction = 0.05

# Launch the graph
with tf.Session(config=config) as sess:
    sess.run(init_op)

    # ckpt = tf.train.get_checkpoint_state('../models/CNN20171110163918')
    # saver.restore(sess, ckpt.model_checkpoint_path)

    start = global_step.eval()
    step = 1
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    while step <= training_iters:
        op = sess.run(optimizer, feed_dict={keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={keep_prob: 1.})
            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " +
                  "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
        if step in save_step:
            global_step.assign(step).eval()
            saver.save(sess, ckpt_dir + '/model.ckpt', global_step=global_step)
        step += 1
    print("Optimization Finished!")
    coord.request_stop()
    coord.join(threads)
