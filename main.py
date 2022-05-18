import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable tensorflow warnings

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import datetime
from vae.model import VariationalAutoencoder
from vae.utils import create_dataset, plot_encoding, plot_reconstruction


def main():

    # set random seed

    tf.random.set_seed(42)

    # define models parameters

    units = [128, 64, 32]
    latent_dim = 2
    input_dim = 28 * 28
    beta = 10.

    # create optimizer

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    # create model
    model = VariationalAutoencoder(input_dim=input_dim, latent_dim=latent_dim, units=units, beta=beta)

    # compile model

    model.compile(optimizer=optimizer)

    # create dataset

    train_dataset, (test_images, test_labels) = create_dataset()

    # create tensorboard callback

    log_dir = './logs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = TensorBoard(log_dir=log_dir)

    # train model

    model.fit(train_dataset, epochs=100, callbacks=[tensorboard])

    # plot encodings

    encoding, _ = model.encode(test_images)
    plot_encoding(encoding, test_labels, save_dir='./results')

    # plot reconstruction

    reconstruction = model.predict(test_images)
    plot_reconstruction(reconstruction, test_labels, save_dir='./results')


if __name__ == '__main__':
    main()
