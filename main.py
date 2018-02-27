import os.path

import scipy
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'

    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    input_tensor = sess.graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob_tensor = sess.graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    vgg_layer3_out_tensor = sess.graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    vgg_layer4_out_tensor = sess.graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    vgg_layer7_out_tensor = sess.graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return input_tensor, keep_prob_tensor, vgg_layer3_out_tensor, vgg_layer4_out_tensor, vgg_layer7_out_tensor
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    output = tf.layers.conv2d(vgg_layer7_out, num_classes, 1,
                              strides=(1, 1),
                              kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    output = tf.layers.conv2d_transpose(output, num_classes, 4,
                                        strides=(2, 2),
                                        padding='same',
                                        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    tf.Print(output, [tf.shape(output)])

    output_l4 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1,
                                 padding='same',
                                 kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    output = tf.add(output, output_l4)

    output = tf.layers.conv2d_transpose(output,
                                        num_classes,
                                        4,
                                        strides=(2, 2),
                                        padding='same',
                                        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    tf.Print(output, [tf.shape(output)])

    output_l3 = tf.layers.conv2d(vgg_layer3_out,
                                 num_classes,
                                 1,
                                 strides=(1, 1),
                                 kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    output = tf.add(output, output_l3)

    output = tf.layers.conv2d_transpose(output, num_classes, 16,
                                        strides=(8, 8),
                                        padding='same',
                                        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    tf.Print(output, [tf.shape(output)])

    return output
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1, num_classes))

    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))

    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    cross_entropy_loss = cross_entropy_loss + 0.001 * tf.reduce_sum(reg_losses)

    adam_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy_loss) # + tf.reduce_sum(reg_losses))

    return logits, adam_optimizer, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    step = 0

    for epoch in range(epochs):
        print("epoch {}".format(epoch))

        for image, label in get_batches_fn(batch_size):
            step += 1
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict={input_image: image, correct_label: label, keep_prob: 0.5, learning_rate: 0.0005})

            if step % 10 == 0:
                print("cross_entropy at step {} : {}".format(step, loss))


tests.test_train_nn(train_nn)


def train():
    num_classes = 2
    image_shape = (160, 576)
    epochs = flags.FLAGS.epochs
    batch_size = flags.FLAGS.batch_size
    data_dir = flags.FLAGS.data_path
    runs_dir = flags.FLAGS.run_path

    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # TODO: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # TODO: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        layer_output = layers(layer3_out, layer4_out, layer7_out, num_classes)

        # Train NN using the train_nn function
        correct_label = tf.placeholder(tf.float32, [None, None, None, num_classes])
        learning_rate = tf.placeholder(tf.float32)

        logits, adam_optimizer, cross_entropy_loss = optimize(layer_output, correct_label, learning_rate, num_classes)

        sess.run(tf.global_variables_initializer())

        # to save the trained model (preparation)
        saver = tf.train.Saver()

        if flags.FLAGS.use_snapshot:
            # restore a saved model here:
            saver.restore(sess, tf.train.latest_checkpoint('./runs/'))

        train_nn(sess, epochs, batch_size, get_batches_fn, adam_optimizer, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate)

        saver.save(sess, 'trained_model.ckpt', global_step=1000, write_meta_graph=False)

        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)


def restore_graph(sess, runs_dir):
    sess.run(tf.global_variables_initializer())

    #restore a saved model here:
    saver = tf.train.import_meta_graph(os.path.join(runs_dir, 'trained_model.ckpt.meta'))
    saver.restore(sess, tf.train.latest_checkpoint(runs_dir))

    graph = sess.graph
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    input_image = graph.get_tensor_by_name('image_input:0')

    if flags.FLAGS.verbose:
        for op in tf.get_default_graph().get_operations():
            print(str(op.name))

    logits = graph.get_tensor_by_name('Reshape:0')
    return keep_prob, logits, input_image


def test():
    data_dir = flags.FLAGS.data_path
    runs_dir = flags.FLAGS.run_path

    with tf.Session() as sess:
        keep_prob, logits, input_image = restore_graph(sess, runs_dir)
        image_shape = (160, 576)

        import numpy as np

        def process_image(image):
            image = np.asarray(Image.fromarray(image), dtype=np.uint8)
            image = scipy.misc.imresize(image, image_shape)

            im_softmax = sess.run(
                [tf.nn.softmax(logits)],
                {keep_prob: 1.0, input_image: [image]})

            im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
            segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)

            mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
            mask = scipy.misc.toimage(mask, mode="RGBA")

            street_im = scipy.misc.toimage(image)
            street_im.paste(mask, box=None, mask=mask)

            return np.array(street_im)

        from moviepy.editor import VideoFileClip
        from PIL import Image

        # clip = VideoFileClip('driving.mp4').subclip(0, 5)
        clip = VideoFileClip(os.path.join(data_dir, 'driving.mp4'))

        new_clip = clip.fl_image(process_image)

        new_clip.write_videofile(os.path.join(runs_dir, 'result.mp4'))


flags = tf.app.flags
flags.DEFINE_enum(name='mode',
                  default='train',
                  enum_values=['train', 'test', 'dry-run'],
                  help='run mode: default is train, use test in order to process sample video file')
flags.DEFINE_bool(name='use_snapshot',
                  default=False,
                  help='Whether to load saved snapshot when run in training mode')

flags.DEFINE_bool(name='verbose',
                  default=False,
                  help='Enable verbose mode')

flags.DEFINE_string(name='data_path',
                    default='./data',
                    help='Data path to load vgg model and train dataset')

flags.DEFINE_string(name='run_path',
                    default='./runs',
                    help='Path to store output data')

flags.DEFINE_integer(name='epochs',
                     default=20,
                     help='Number of epochs for training')

flags.DEFINE_integer(name='batch_size',
                     default=16,
                     help='Batch size used for training')


def main(args):
    if flags.FLAGS.mode == 'train':
        train()
    elif flags.FLAGS.mode == 'test':
        test()
    else:
        print(flags.FLAGS.main_module_help())

if __name__ == '__main__':
    tf.app.run()

