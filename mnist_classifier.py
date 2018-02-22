from __future__ import print_function

import os
import sys
import numpy as np
import datetime
import dateutil.tz

import tensorflow as tf
import tensorflow.contrib.layers as tcl
from tensorflow.examples.tutorials.mnist import input_data

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing import image


flags = tf.app.flags
flags.DEFINE_integer("epochs", 50, "Epochs to train [25]")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate for the optimizer [0.0002]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_string("optimizer", "Adam", "Optimizer to use [Adadelta]")
flags.DEFINE_integer("val_size", 5000, "The size of the validation set [5000]")
flags.DEFINE_string("log_dir", "log_dir", "Directory name to save the checkpoints and logs []")
flags.DEFINE_boolean("train", False, "True for training a new model [False]")
flags.DEFINE_boolean("predict", False, "True for predicting with an existing model [False]")
flags.DEFINE_string("data_set_path", "mnist_data", "Path where data set is stored. [mnist_data]")
flags.DEFINE_string("model", "weights.hdf5", "Path to model used for prediction. []")
flags.DEFINE_string("img_path", "", "Path to images to predict. []")
FLAGS = flags.FLAGS

now = datetime.datetime.now(dateutil.tz.tzlocal())
timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

log_dir = FLAGS.log_dir+"/" + str(sys.argv[0][:-3]) + "_" + timestamp
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


# save command line arguments
with open(log_dir + "/hyperparameters_"+timestamp+".csv", "wb") as f:
    for arg in tf.app.flags.FLAGS.flag_values_dict():
        f.write(arg + "," + str(tf.app.flags.FLAGS.flag_values_dict()[arg]) + "\n")


# use mnist data from the specified folder (download if not already there)
def load_mnist_data(path, val_size):
    mnist = input_data.read_data_sets(path, validation_size=val_size, one_hot=True)

    x_train = mnist.train.images
    x_train = np.reshape(x_train, (-1, 28, 28, 1))
    y_train = mnist.train.labels

    x_val = mnist.validation.images
    x_val = np.reshape(x_val, (-1, 28, 28, 1))
    y_val = mnist.validation.labels

    x_test = mnist.test.images
    x_test = np.reshape(x_test, (-1, 28, 28, 1))
    y_test = mnist.test.labels

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


# build the classification model
def build_model(optimizer, learning_rate, input_shape=(28, 28, 1)):
    weight_init = tf.random_normal_initializer(stddev=0.02)
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(4, 4), input_shape=input_shape, padding="same", strides=2,
                     kernel_initializer=weight_init))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Dropout(0.5))

    model.add(Conv2D(128, kernel_size=(4, 4), padding="same", strides=2, kernel_initializer=weight_init))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Dropout(0.5))

    model.add(Conv2D(256, kernel_size=(4, 4), padding="same", strides=2, kernel_initializer=weight_init))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Dropout(0.5))

    model.add(Conv2D(256, kernel_size=(4, 4), padding="same", strides=2, kernel_initializer=weight_init))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(1024, kernel_initializer=weight_init))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.5))

    model.add(Dense(10, activation='softmax'))

    lr = learning_rate
    optimizers = {"SGD": keras.optimizers.SGD(lr=lr), "RMSprop": keras.optimizers.RMSprop(lr=lr),
                  "Adadelta": keras.optimizers.Adadelta(lr=lr), "Adam": keras.optimizers.Adam(lr=lr)
                  }

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=optimizers[optimizer],
                  metrics=['accuracy'])

    return model


# training the model
def train_model():
    train_data, val_data, test_data = load_mnist_data(path=FLAGS.data_set_path, val_size=FLAGS.val_size)

    model = build_model(optimizer=FLAGS.optimizer, learning_rate=FLAGS.learning_rate)

    # callback for the training process
    save_model = keras.callbacks.ModelCheckpoint(log_dir+"/weights.hdf5", monitor='val_acc', mode='max', verbose=0,
                                                 save_best_only=True, save_weights_only=False, period=1)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=5, verbose=0, mode='max')
    tensorboard = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=10, batch_size=32, write_graph=True,
                                              write_grads=False, write_images=False, embeddings_freq=0,
                                              embeddings_layer_names=None, embeddings_metadata=None)

    # train model
    model.fit(train_data[0], train_data[1],
              batch_size=FLAGS.batch_size,
              epochs=FLAGS.epochs,
              verbose=1,
              validation_data=val_data,
              callbacks=[early_stopping, save_model, tensorboard])

    # calculate and store test set performance on the model with best validation error
    print("Calculating performance on test set...")
    model = keras.models.load_model(log_dir+"/weights.hdf5")
    score = model.evaluate(test_data[0], test_data[1], verbose=0)
    print('Test loss: {:.4f}'.format(score[0]))
    print('Test accuracy: {:.4f}'.format(score[1]))
    with open(log_dir+"/test_acc-{:.4f}_test_loss-{:.4f}.txt".format(score[1], score[0]), "wb") as file:
        file.write('Test accuracy: {:.4f}\n'.format(score[1]))
        file.write('Test loss: {:.4f}'.format(score[0]))


# predict image classes
def predict(model, img_path, batch_size):
    # helper function to normalize image pixel values into range [0,1]
    def normalize_image(img):
        return img * 1.0 / 255.0

    model = keras.models.load_model(model)
    img_generator = image.ImageDataGenerator(preprocessing_function=normalize_image)
    validation_generator = img_generator.flow_from_directory(directory=img_path, target_size=(28,28), shuffle=False,
                                                             batch_size=batch_size, color_mode="grayscale",)

    score = model.evaluate_generator(validation_generator)
    print("Accuracy: {:.4f}".format(score[1]))


if FLAGS.train:
    train_model()
elif FLAGS.predict:
    predict(FLAGS.model, FLAGS.img_path, FLAGS.batch_size)
else:
    print("No valid option chosen. Choose either \"--train\" or \"--predict\".")