import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #  disable tensorflow warnings

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def preprocess_images(images):
    """ Reshape and rescale images. """
    images = tf.reshape(images, shape=(tf.shape(images)[0], 28 * 28))
    images = tf.cast(images, tf.dtypes.float32) / 255.
    return images


def create_labels(inputs):
    """ Create input output pairs"""
    return inputs, inputs


def create_dataset(batch=32):
    """ Create dataset. """
    (train_images, _), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    train_size = train_images.shape[0]
    test_size = test_images.shape[0]

    train_dataset = tf.data.Dataset.from_tensor_slices(train_images)
    train_dataset = train_dataset.shuffle(train_size).batch(batch).map(preprocess_images).map(create_labels)

    test_images = test_images.reshape(-1, 28 * 28).astype('float32') / 255.

    return train_dataset, (test_images, test_labels)


def plot_encoding(inputs, labels, save_dir=None):
    """ Plot encodings. """
    plt.figure()
    plt.title('encodings')
    plt.scatter(inputs[:, 0], inputs[:, 1], c=labels)
    plt.colorbar()
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'encodings.png'))
    plt.show()


def plot_reconstruction(inputs, labels, save_dir=None):
    """ Plot reconstruction. """
    idx = np.random.randint(low=0, high=inputs.shape[0], size=16)
    inputs = inputs[idx]
    labels = labels[idx]
    plt.figure()
    for i, (image, label) in enumerate(zip(inputs, labels)):
        plt.subplot(4, 4, i + 1)
        plt.title(str(label))
        image = image.reshape((28, 28))
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'reconstruction.png'))
    plt.show()

