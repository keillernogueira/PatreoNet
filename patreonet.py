import datetime
import numpy as np
import scipy.misc
import tensorflow as tf
import sys
import os
import random

from os import listdir
from PIL import Image
from skimage import img_as_float
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from tensorflow.python.framework import ops


NUM_CLASSES = 21


class BatchColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


'''
Utils
'''


def print_params(list_params):
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    for i in range(1, len(sys.argv)):
        print(list_params[i - 1] + '= ' + sys.argv[i])
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')


def select_batch(shuffle, batch_size, it, total_size):
    batch = shuffle[it:min(it + batch_size, total_size)]
    if min(it + batch_size, total_size) == total_size or total_size == it + batch_size:
        shuffle = np.asarray(random.sample(range(total_size), total_size))
        it = 0
        if len(batch) < batch_size:
            diff = batch_size - len(batch)
            batch_c = shuffle[it:it + diff]
            batch = np.concatenate((batch, batch_c))
            it = diff
    else:
        it += batch_size
    return shuffle, batch, it


'''
Image Processing
'''


def normalize_images(data, mean_full, std_full):
    data[:, :, :] = np.subtract(data[:, :, :], mean_full)
    data[:, :, :] = np.divide(data[:, :, :], std_full)


def compute_image_mean(data):
    mean_full = np.mean(np.mean(np.mean(data, axis=0), axis=0), axis=0)
    std_full = np.std(data, axis=0, ddof=1)[0, 0]
    return mean_full, std_full


def load_images(dataset_path, data_augmentation=True, resize_to=224):
    data = []
    classes = []

    for d in listdir(dataset_path):
        for f in listdir(dataset_path + d):
            img = Image.open(dataset_path + d + '/' + f)
            img_resize = img.resize((resize_to, resize_to), Image.ANTIALIAS)
            img_resize.load()
            img_float = img_as_float(img_resize)
            data.append(img_float)
            classes.append(d)
            # DATA AUGMENTATION
            if data_augmentation is True:
                data.append(np.fliplr(img_float))
                classes.append(d)

    le = preprocessing.LabelEncoder()
    le.fit(classes)
    classes_code = le.transform(classes)

    data_arr = np.asarray(data)
    classes_arr = np.asarray(classes_code)
    print(data_arr.shape, classes_arr.shape)
    print(np.bincount(classes_arr.flatten()))

    return data_arr, classes_arr


'''
TensorFlow
'''


def leaky_relu(x, alpha=0.1):
    return tf.maximum(alpha * x, x)


def _variable_on_cpu(name, shape, ini):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=ini, dtype=tf.float32)
    return var


def _variable_with_weight_decay(name, shape, ini, weight_decay):
    var = _variable_on_cpu(name, shape, ini)
    # tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32)
    # tf.contrib.layers.xavier_initializer(dtype=tf.float32))
    # tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
    if weight_decay is not None:
        try:
            weight_decay = tf.mul(tf.nn.l2_loss(var), weight_decay, name='weight_loss')
        except:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), weight_decay, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def _batch_norm(input_data, is_training, scope=None):
    # Note: is_training is tf.placeholder(tf.bool) type
    return tf.cond(is_training,
                   lambda: tf.contrib.layers.batch_norm(input_data, is_training=True, center=False,
                                                        updates_collections=None,
                                                        scope=scope),
                   lambda: tf.contrib.layers.batch_norm(input_data, is_training=False, center=False,
                                                        updates_collections=None, scope=scope, reuse=True)
                   )


def _conv_layer(input_data, kernel_shape, name, weight_decay, is_training, strides=None, pad='SAME',
                activation=None, norm=None):
    if strides is None:
        strides = [1, 1, 1, 1]
    with tf.variable_scope(name) as scope:
        weights = _variable_with_weight_decay('weights', shape=kernel_shape,
                                              ini=tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32),
                                              weight_decay=weight_decay)
        biases = _variable_on_cpu('biases', kernel_shape[-1], tf.constant_initializer(0.1))

        conv_op = tf.nn.conv2d(input_data, weights, strides, padding=pad)
        conv_op = tf.nn.bias_add(conv_op, biases)

        if norm == 'batch_norm':
            conv_op = _batch_norm(conv_op, is_training, scope=scope)
        elif norm == 'lrn':
            conv_op = tf.nn.local_response_normalization(conv_op)

        if activation == 'LeakyReLU':
            conv_op = leaky_relu(conv_op)
        elif activation == 'relu':
            conv_op = tf.nn.relu(conv_op)

        return conv_op


def maxpool2d(x, k, strides=1, pad='SAME', name=None):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, strides, strides, 1], padding=pad, name=name)


def patreo_net(x, keep_prob, is_training, input_data_image_size, weight_decay):
    x = tf.reshape(x, shape=[-1, input_data_image_size, input_data_image_size, 3])  # RGB

    # layer 1
    conv1 = _conv_layer(x, [4, 4, 3, 96], 'conv1', weight_decay, is_training, strides=[1, 3, 3, 1], pad='VALID',
                        activation='relu', norm='lrn')
    pool1 = maxpool2d(conv1, k=2, strides=2, pad='VALID', name='pool1')

    # layer 2
    conv2 = _conv_layer(pool1, [4, 4, 96, 256], 'conv2', weight_decay, is_training, strides=[1, 2, 2, 1], pad='VALID',
                        activation='relu', norm='lrn')
    pool2 = maxpool2d(conv2, k=2, strides=2, pad='VALID', name='pool2')

    # layer 3
    conv3 = _conv_layer(pool2, [2, 2, 256, 256], 'conv3', weight_decay, is_training, strides=[1, 1, 1, 1], pad='VALID',
                        activation='relu')
    pool3 = maxpool2d(conv3, k=2, strides=2, pad='VALID', name='pool3')

    # Fully connected layer 1
    with tf.variable_scope('fc6') as scope:
        # b, w, h, c = pool3.get_shape().as_list()
        reshape = tf.reshape(pool3, [-1, 3*3*256])

        weights = _variable_with_weight_decay('weights', shape=[3 * 3 * 256, 1024],
                                              ini=tf.contrib.layers.xavier_initializer(dtype=tf.float32),
                                              weight_decay=weight_decay)
        biases = _variable_on_cpu('biases', [1024], tf.constant_initializer(0.1))

        # Apply Dropout
        drop_fc1 = tf.nn.dropout(reshape, keep_prob)
        fc1 = tf.nn.relu(tf.add(tf.matmul(drop_fc1, weights), biases))

    # Fully connected layer 2
    with tf.variable_scope('fc7') as scope:
        weights = _variable_with_weight_decay('weights', shape=[1024, 1024],
                                              ini=tf.contrib.layers.xavier_initializer(dtype=tf.float32),
                                              weight_decay=weight_decay)
        biases = _variable_on_cpu('biases', [1024], tf.constant_initializer(0.1))

        # Apply Dropout
        drop_fc2 = tf.nn.dropout(fc1, keep_prob)
        fc2 = tf.nn.relu(tf.add(tf.matmul(drop_fc2, weights), biases))

    # Output, class prediction
    with tf.variable_scope('init_softmax') as scope:
        weights = _variable_with_weight_decay('weights', [1024, NUM_CLASSES],
                                              ini=tf.contrib.layers.xavier_initializer(dtype=tf.float32),
                                              weight_decay=weight_decay)
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.1))
        logits = tf.add(tf.matmul(fc2, weights), biases, name=scope.name)

    return logits


def loss_def(logits, labels):
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels,
                                                                   name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def validate(sess, data, classes, n_input_data, batch_size, x, y, keep_prob, is_training, pred, acc_mean, step):
    all_predcs = []
    cm_test = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.uint32)
    true_count = 0.0

    for i in range(0, ((len(classes) / batch_size) + 1 if len(classes) % batch_size != 0 else
                        (len(classes) / batch_size))):
        bx = np.reshape(data[i * batch_size:min(i * batch_size + batch_size, len(classes))], (-1, n_input_data))
        by = classes[i * batch_size:min(i * batch_size + batch_size, len(classes))]

        preds_val, acc_mean_val = sess.run([pred, acc_mean], feed_dict={x: bx, y: by, keep_prob: 1., is_training: False})
        true_count += acc_mean_val

        all_predcs = np.concatenate((all_predcs, preds_val))

        for j in range(len(preds_val)):
            cm_test[by[j]][preds_val[j]] += 1

    _sum = 0.0
    for i in range(len(cm_test)):
        _sum += (cm_test[i][i] / float(np.sum(cm_test[i])) if np.sum(cm_test[i]) != 0 else 0)

    print("---- Iter " + str(step) + " -- Time " + str(datetime.datetime.now().time()) +
          " -- Test: Overall Accuracy= " + "{:.6f}".format(true_count / float(len(classes))) +
          " Normalized Accuracy= " + "{:.6f}".format(_sum / float(NUM_CLASSES)) +
          " Confusion Matrix= " + np.array_str(cm_test).replace("\n", "")
          )


def train(data_train, class_train, data_test, class_test, model_path,
          x, y, keep_prob, dropout, is_training, n_input_data, batch_size, niter,
          optimizer, loss, acc_mean, pred, output_path):
    ###################
    epoch_number = 1680  # int(len(data_train)/batch_size) # 1 epoch = images / batch
    val_inteval = 1680  # int(len(data_train)/batch_size)
    display_step = 50  # math.ceil(int(len(training_classes)/batch_size)*0.01)
    ###################

    # Initializing the variables
    init = tf.initialize_all_variables()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    saver_restore = tf.train.Saver()
    current_iter = 1

    shuffle = np.asarray(random.sample(range(len(class_train)), len(class_train)))

    # Launch the graph
    with tf.Session() as sess:
        if 'model' in model_path:
            try:
                current_iter = int(model_path.split('_')[-1])
            except:
                current_iter = int(model_path.split('-')[-1])
            print BatchColors.OKBLUE + 'Model restored from ' + model_path + BatchColors.ENDC
            saver_restore.restore(sess, model_path)
        else:
            sess.run(init)
            print BatchColors.OKBLUE + 'Model totally initialized!' + BatchColors.ENDC

        it = 0
        epoch_mean = 0.0
        epoch_cm_train = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.uint32)
        batch_cm_train = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.uint32)

        # Keep training until reach max iterations
        for step in range(current_iter, niter + 1):
            shuffle, batch, it = select_batch(shuffle, batch_size, it, len(class_train))
            data = data_train[batch]
            batch_y = class_train[batch]
            batch_x = np.reshape(data, (-1, n_input_data))

            _, batch_loss, batch_correct, batch_predcs = sess.run([optimizer, loss, acc_mean, pred],
                                                                  feed_dict={x: batch_x, y: batch_y,
                                                                             keep_prob: dropout, is_training: True})

            epoch_mean += batch_correct
            for j in range(len(batch_predcs)):
                epoch_cm_train[batch_y[j]][batch_predcs[j]] += 1

            if step % display_step == 0:
                # Calculate batch loss and accuracy
                for j in range(len(batch_predcs)):
                    batch_cm_train[batch_y[j]][batch_predcs[j]] += 1

                _sum = 0.0
                for i in range(len(batch_cm_train)):
                    _sum += (batch_cm_train[i][i] / float(np.sum(batch_cm_train[i])) if np.sum(
                        batch_cm_train[i]) != 0 else 0)

                print("Iter " + str(step) + " -- Training Minibatch: Loss= " + "{:.6f}".format(batch_loss) +
                      " Absolut Right Pred= " + str(int(batch_correct)) +
                      " Overall Accuracy= " + "{:.4f}".format(batch_correct / float(len(batch_y))) +
                      " Normalized Accuracy= " + "{:.4f}".format(_sum / float(NUM_CLASSES)) +
                      " Confusion Matrix= " + np.array_str(batch_cm_train).replace("\n", "")
                      )
                batch_cm_train = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.uint32)

            # if Conv_net_type == 'mph' and step % 25000 == 0:
            # save_feature_map(_pool,_open,_close, output_path+'fold'+str(fold)+'/', step)

            if step % epoch_number == 0:
                _sum = 0.0
                for i in range(len(epoch_cm_train)):
                    _sum += (epoch_cm_train[i][i] / float(np.sum(epoch_cm_train[i])) if np.sum(
                        epoch_cm_train[i]) != 0 else 0)

                print("Iter " + str(step) + " -- Training Epoch:" +
                      " Overall Accuracy= " + "{:.6f}".format(epoch_mean / float(len(class_train))) +
                      " Normalized Accuracy= " + "{:.6f}".format(_sum / float(NUM_CLASSES)) +
                      " Confusion Matrix= " + np.array_str(epoch_cm_train).replace("\n", "")
                      )

                epoch_mean = 0.0
                epoch_cm_train = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.uint32)

            if step % val_inteval == 0:
                # Test
                saver.save(sess, output_path + 'model', global_step=step)

                # scipy.misc.imsave(output_path + 'fold' + str(fold) + '/' + str(step) + '_maxout1.jpeg',
                #                  max1_out[0, :, :, 0])
                validate(sess, data_test, class_test, n_input_data, batch_size, x, y, keep_prob, is_training, pred,
                         acc_mean, step)

        print("Optimization Finished!")

        # Test: Final
        saver.save(sess, output_path + 'model', global_step=step)
        validate(sess, data_test, class_test, n_input_data, batch_size, x, y, keep_prob, is_training, pred,
                 acc_mean, step)


def test(data_test, class_test, model_path,
         x, y, keep_prob, is_training, n_input_data, batch_size,
         acc_mean, pred):
    saver_restore = tf.train.Saver()

    # Launch the graph
    with tf.Session() as sess:
        try:
            step = int(model_path.split('_')[-1])
        except:
            step = int(model_path.split('-')[-1])
        print BatchColors.OKBLUE + 'Model restored from ' + model_path + BatchColors.ENDC
        saver_restore.restore(sess, model_path)

        validate(sess, data_test, class_test, n_input_data, batch_size, x, y, keep_prob, is_training, pred,
                 acc_mean, step)


'''
python patreonet.py /home/UCMerced/Images/ /home/aux/ /home/aux/ 0.01 0.005 256 200000 1 train
'''


def main():
    list_params = ['dataset_path', 'output_path(for model, images, etc)', 'model_path (to load)',
                   'learning_rate', 'weight_decay', 'batch_size', 'niter', 
                   'fold', 'operation [train|test]']
    if len(sys.argv) < len(list_params) + 1:
        sys.exit('Usage: ' + sys.argv[0] + ' ' + ' '.join(list_params))
    print_params(list_params)

    index = 1
    dataset_path = sys.argv[index]
    index = index + 1
    output_path = sys.argv[index]
    index = index + 1
    model_path = sys.argv[index]

    # Parameters
    index = index + 1
    lr_initial = float(sys.argv[index])
    index = index + 1
    weight_decay = float(sys.argv[index])
    index = index + 1
    batch_size = int(sys.argv[index])
    index = index + 1
    niter = int(sys.argv[index])

    index = index + 1
    fold = int(sys.argv[index])
    index = index + 1
    operation = sys.argv[index]
    input_data_image_size = 224

    # LOAD IMAGES
    data, classes = load_images(dataset_path, data_augmentation=False, resize_to=input_data_image_size)

    if os.path.isfile(os.path.join(os.getcwd(), 'test_fold' + str(fold) + '.npy')):
        training_distribution = np.load(os.path.join(os.getcwd(), 'train_fold' + str(fold) + '.npy'))
        test_distribution = np.load(os.path.join(os.getcwd(), 'test_fold' + str(fold) + '.npy'))
        print(BatchColors.OKGREEN + "Fold " + str(fold) + " loaded!" + BatchColors.ENDC)
    else:
        print(BatchColors.WARNING + "Could not locate old folds! New ones will be generated" + BatchColors.ENDC)
        print(BatchColors.OKBLUE + 'Creating class distribution...' + BatchColors.ENDC)

        skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=True)
        f = 1
        for train_index, test_index in skf.split(data, classes):
            np.save(os.path.join(os.getcwd(), 'train_fold' + str(f) + '.npy'), train_index)
            np.save(os.path.join(os.getcwd(), 'test_fold' + str(f) + '.npy'), test_index)
            f += 1
        print(BatchColors.WARNING + "Folds saved!" + BatchColors.ENDC)
        print(BatchColors.WARNING + "Suggest to RUN all again with new folds!" + BatchColors.ENDC)
        print(BatchColors.WARNING + "Running fold " + str(fold) + BatchColors.ENDC)
        training_distribution = np.load(os.path.join(os.getcwd(), 'train_fold' + str(fold) + '.npy'))
        test_distribution = np.load(os.path.join(os.getcwd(), 'test_fold' + str(fold) + '.npy'))

    data_train, data_test = data[training_distribution], data[test_distribution]
    class_train, class_test = classes[training_distribution], classes[test_distribution]

    print(data_train.shape, class_train.shape)
    print(data_test.shape, class_test.shape)

    mean_full, std_full = compute_image_mean(data_train)
    normalize_images(data_train, mean_full, std_full)
    normalize_images(data_test, mean_full, std_full)

    # Network Parameters
    n_input_data = input_data_image_size * input_data_image_size * 3  # RGB
    dropout = 0.5  # Dropout, probability to keep units

    # tf Graph input_data
    x = tf.placeholder(tf.float32, [None, n_input_data])
    y = tf.placeholder(tf.int32, [None])
    keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)
    is_training = tf.placeholder(tf.bool, [], name='is_training')
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # CONVNET
    logits = patreo_net(x, keep_prob, is_training, input_data_image_size, weight_decay)

    # Define loss and optimizer
    loss = loss_def(logits, y)
    lr = tf.train.exponential_decay(lr_initial, global_step, 50000, 0.1, staircase=True)
    optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(loss, global_step=global_step)

    # Evaluate model
    correct = tf.nn.in_top_k(logits, y, 1)
    # Return the number of true entries
    acc_mean = tf.reduce_sum(tf.cast(correct, tf.int32))
    pred = tf.argmax(logits, 1)
    # correct_pred = tf.equal(pred, tf.argmax(y, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    if operation == 'train':
        train(data_train, class_train, data_test, class_test, model_path,
              x, y, keep_prob, dropout, is_training, n_input_data, batch_size, niter,
              optimizer, loss, acc_mean, pred, output_path)
    elif operation == 'test':
        test(data_test, class_test, model_path,
             x, y, keep_prob, is_training, n_input_data, batch_size,
             acc_mean, pred)
    else:
        print BatchColors.FAIL + "Operation not found: " + operation + BatchColors.ENDC
        print BatchColors.WARNING + "Options are: train or test" + BatchColors.ENDC


if __name__ == "__main__":
    main()
