#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

from timeit import default_timer as timer


# ---------------------------------
#       Hyper Parameters
# ---------------------------------
L2_REG = 1e-3
STDEV = 1e-2
KEEP_PROB = 0.8
LEARNING_RATE = 1e-4

EPOCHS = 30
BATCH_SIZE = 8
IMAGE_SHAPE = (160, 576)
NUM_CLASSES = 2

# ---------------------------------
#       Check Tensor Flow
# ---------------------------------
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# ---------------------------------
#       Check GPU
# ---------------------------------
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


# ---------------------------------
#       Load VGG
# ---------------------------------
def load_vgg(sess, vgg_path):

    # Name Tensors
    vgg_tag = 'vgg16'

    # Load Graph
    graph = tf.get_default_graph()

    # Load Model
    tf.saved_model.load(sess, [vgg_tag], vgg_path)

    # Get Tensors by Name
    w1 = graph.get_tensor_by_name('image_input:0')
    keep = graph.get_tensor_by_name('keep_prob:0')
    layer3_out = graph.get_tensor_by_name('layer3_out:0')
    layer4_out = graph.get_tensor_by_name('layer4_out:0')
    layer7_out = graph.get_tensor_by_name('layer7_out:0')

    return w1, keep, layer3_out, layer4_out, layer7_out


# tests.test_load_vgg(load_vgg, tf)

# ---------------------------------
#       Layers
# ---------------------------------
def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):

    init = tf.truncated_normal_initializer(stddev=STDEV)
    reg = tf.contrib.layers.l2_regularizer(L2_REG)

    # Convolutions
    conv_1x1_7 = tf.layers.conv2d(
        inputs=vgg_layer7_out,
        kernel_size=1,
        strides=1,
        padding='same',
        filters=num_classes,
        kernel_initializer=init,
        kernel_regularizer=reg
    )

    conv_1x1_4 = tf.layers.conv2d(
        inputs=vgg_layer4_out,
        kernel_size=1,
        strides=1,
        padding='same',
        filters=num_classes,
        kernel_initializer=init,
        kernel_regularizer=reg
    )

    conv_1x1_3 = tf.layers.conv2d(
        inputs=vgg_layer3_out,
        kernel_size=1,
        strides=1,
        padding='same',
        filters=num_classes,
        kernel_initializer=init,
        kernel_regularizer=reg
    )

    # Up-sample
    up_conv_1 = tf.layers.conv2d_transpose(
        inputs=conv_1x1_7,
        kernel_size=4,
        strides=2,
        padding='same',
        filters=num_classes,
        kernel_initializer=init,
        kernel_regularizer=reg
    )

    skip_1 = tf.add(up_conv_1, conv_1x1_4)

    up_conv_2 = tf.layers.conv2d_transpose(
        inputs=skip_1,
        kernel_size=4,
        strides=2,
        padding='same',
        filters=num_classes,
        kernel_initializer=init,
        kernel_regularizer=reg
    )

    skip_2 = tf.add(up_conv_2, conv_1x1_3)

    # Put it all together
    upsampled_prediction = tf.layers.conv2d_transpose(
        inputs=skip_2,
        kernel_size=16,
        strides=8,
        padding='same',
        filters=num_classes,
        kernel_initializer=init,
        kernel_regularizer=reg
    )

    return upsampled_prediction


tests.test_layers(layers)


# ---------------------------------
#       Optimize
# ---------------------------------
def optimize(nn_last_layer, correct_label, learning_rate, num_classes):

    # Labels and Logits
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))

    # Loss
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    cross_entropy_loss = tf.reduce_mean(loss)

    # Optimize
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # Train
    train_op = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss


tests.test_optimize(optimize)


# ---------------------------------
#       Train NN
# ---------------------------------
def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):

    for epoch in range(epochs):

        # Set Variables to Zero
        loss = None
        s_time = timer()

        # Process Images
        for image, labels in get_batches_fn(batch_size):
            _, loss = sess.run(
                [train_op, cross_entropy_loss],
                feed_dict={input_image:     image,
                           correct_label:   labels,
                           keep_prob:       KEEP_PROB,
                           learning_rate:   LEARNING_RATE}
            )

        # Print Results
        print("[Epoch: {0}/{1} Loss: {2:4f} Time: {3}]".format(epoch, epochs, loss, (timer() - s_time)))

        # Avoid Over Training
        if loss <= 0.016:
            break


tests.test_train_nn(train_nn)

# ======================================
#           RUN
# ======================================
def run():

    # Path to Data
    data_dir = './data'
    runs_dir = './runs'

    # Test for data
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    print(" -- Start Training --")
    tf.reset_default_graph()

    with tf.Session() as sess:

        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')

        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), IMAGE_SHAPE)

        # Load Pre Trained Model
        input_image, keep_prob, layer3, layer4, layer7 = load_vgg(sess, vgg_path=vgg_path)

        # Build FCN-8
        output_layer = layers(layer3, layer4, layer7, num_classes=NUM_CLASSES)
        correct_label = tf.placeholder(dtype=tf.float32, shape=(None, None, None, NUM_CLASSES), name='correct_label')
        learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')

        # Optimize
        logits, train_op, cross_entropy_loss = optimize(output_layer, correct_label, learning_rate, NUM_CLASSES)

        # Training
        sess.run(tf.global_variables_initializer())
        train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_op, cross_entropy_loss, input_image, correct_label,
                 keep_prob, learning_rate)

        # Save Images
        helper.save_inference_samples(runs_dir, data_dir, sess, IMAGE_SHAPE, logits, keep_prob, input_image)


if __name__ == '__main__':
    run()







































































