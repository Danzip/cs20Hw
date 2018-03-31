import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import time

import utils
from tensorflow.contrib import slim


# define model

def res_layer(input, filters, kernel_size, strides, name, regulizer_lambda=0.0004):
    conv_out = tf.layers.conv2d(input, filters, kernel_size, strides, "SAME", activation=tf.nn.relu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(regulizer_lambda)
                                )

    if input.shape[-1] == conv_out.shape[-1] and strides == 1:
        resample_input = input
    else:
        resample_input = tf.layers.conv2d(input, filters, 1, strides, "SAME", activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(regulizer_lambda)
                                          )
    return tf.add(resample_input, conv_out, name=name)


def ResBlock(input, kernel_size, res_ch, total_layers_in_block, name):
    with tf.name_scope(name):
        name_id = 1
        net = res_layer(input, res_ch, kernel_size, strides=1, name="res" + str(name_id))
        name_id += 1
        for idx in range(total_layers_in_block - 2):
            net = res_layer(net, res_ch, kernel_size, strides=1, name="res" + str(name_id))
            name_id += 1
        return res_layer(net, res_ch, kernel_size, strides=1, name="res_downsample")


def fc_block(input, name, regulizer_lambda=0.0004):
    net = tf.contrib.layers.fully_connected(input,
                                            1024,
                                            activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                                            weights_regularizer=tf.contrib.layers.l2_regularizer(regulizer_lambda),
                                            biases_initializer=tf.zeros_initializer(),
                                            scope=name)

    return tf.contrib.layers.fully_connected(net,
                                             10,
                                             activation_fn=tf.nn.relu,
                                             weights_initializer=tf.contrib.layers.xavier_initializer(),
                                             weights_regularizer=tf.contrib.layers.l2_regularizer(regulizer_lambda),
                                             biases_initializer=tf.zeros_initializer(),
                                             scope=name + "out")


def model(input):
    net = ResBlock(input, 3, 128, 3, "block1")
    net = ResBlock(net, 3, 256, 4, "block2")
    net = ResBlock(net, 3, 512, 4, "block3")
    net=tf.contrib.layers.flatten(net)
    return fc_block(net, "fc_block")


def convert_to_data(train, test, batch_size, n_train, n_test):
    # create tf Datasets
    train_data = tf.data.Dataset.from_tensor_slices(train)
    test_data = tf.data.Dataset.from_tensor_slices(test)

    # Shuffle *still dont understand exactly what this function does and wether if buffer_size < total examples in data the new
    # dataset is smaller and therefore we've lost precious examples we worked hard to gather*
    shufled_train = train_data.shuffle(n_train)
    shufled_test = train_data.shuffle(n_test)

    # batch
    batched_train = shufled_train.batch(batch_size)
    batched_test = shufled_test.batch(batch_size)
    return batched_train, batched_test


if __name__ == "__main__":
    # Define paramaters for the model
    learning_rate = 0.01
    batch_size = 16
    n_epochs = 30
    n_train = 55000
    n_test = 10000

    # Step 1: Read in data
    mnist_folder = '/home/danziv/PycharmProjects/stanford cs20_HW/stanford-tensorflow-tutorials/examples/data/mnist'
    # utils.download_mnist(mnist_folder)
    train, val, test = utils.read_mnist(mnist_folder, flatten=False)
    # Step 2: convert_to_tf_dataset_api
    batched_train, batched_test = convert_to_data(train, test, batch_size, n_train, n_test)
    # Step 3: create dataset iterators
    iterator = tf.data.Iterator.from_structure(batched_train.output_types,
                                               batched_test.output_shapes)
    image_batch, label_batch = iterator.get_next()

    train_iterator = iterator.make_initializer(batched_train)
    test_iterator = iterator.make_initializer(batched_test)

    # Step 4: define the model
    logits = model(tf.expand_dims(image_batch, axis=-1))

    # Step 5: define loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_batch, logits=logits), name="loss")

    # Step 6: define optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # Step 7: define accuracy calculation
    correct_preds = tf.equal(tf.argmax(logits, axis=1), tf.argmax(label_batch, axis=1))
    ammount_correct = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

    # step 8: define training
    writer = tf.summary.FileWriter('./graphs/res', tf.get_default_graph())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(n_epochs):
            total_loss = 0
            n_batches = 0
            start = time.time()
            sess.run(train_iterator)
            try:
                while True:
                    _, l = sess.run([optimizer, loss])
                    n_batches += 1
                    total_loss += l
                    print('Average loss batch {0}: {1}'.format(n_batches, l))
            except tf.errors.OutOfRangeError:
                pass
            print('Average loss epoch {0}: {1}'.format(epoch, total_loss / n_batches))
            print('Total time epoch {0}: {1} seconds'.format(epoch, time.time() - start))

        sess.run(test_iterator)  # drawing samples from test_data
        total_correct_preds = 0
        try:
            while True:
                accuracy_batch = sess.run(ammount_correct)
                total_correct_preds += accuracy_batch
        except tf.errors.OutOfRangeError:
            pass

        print('Accuracy {0}'.format(total_correct_preds / n_test))
    writer.close()
