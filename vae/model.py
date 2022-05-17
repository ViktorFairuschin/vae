import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #  disable tensorflow warnings

import tensorflow as tf


class Sampling(tf.keras.layers.Layer):
    """ Sampling layer. """

    def __init__(self, **kwargs):
        super(Sampling, self).__init__(**kwargs)

    def call(self, inputs, *args, **kwargs):
        mean, logvar = inputs
        eps = tf.random.normal(shape=tf.shape(mean))
        z = eps * tf.exp(logvar * .5) + mean
        return z

    def get_config(self):
        config = super(Sampling, self).get_config()
        return config


class Encoder(tf.keras.layers.Layer):
    """ Encoder layer. """

    def __init__(
            self,
            latent_dim,
            units,
            **kwargs
    ):
        super(Encoder, self).__init__(**kwargs)

        self.latent_dim = latent_dim
        self.units = units

        self.dense_block = [tf.keras.layers.Dense(units, activation='relu') for units in self.units]
        self.mean = tf.keras.layers.Dense(self.latent_dim)
        self.logvar = tf.keras.layers.Dense(self.latent_dim)

    def call(self, inputs, **kwargs):
        x = inputs
        for layer in self.dense_block:
            x = layer(x)
        mean = self.mean(x)
        logvar = self.logvar(x)
        return mean, logvar

    def get_config(self):
        config = super(Encoder, self).get_config()
        config.update({"latent_dim": self.latent_dim})
        config.update({"units": self.units})
        return config


class Decoder(tf.keras.layers.Layer):
    """ Decoder layer. """

    def __init__(
            self,
            output_dim,
            units,
            **kwargs
    ):
        super(Decoder, self).__init__(**kwargs)

        self.output_dim = output_dim
        self.units = units

        self.dense_block = [tf.keras.layers.Dense(units, 'relu') for units in self.units]
        self.outputs = tf.keras.layers.Dense(self.output_dim)

    def call(self, inputs, **kwargs):
        x = inputs
        for layer in self.dense_block:
            x = layer(x)
        x = self.outputs(x)
        return x

    def get_config(self):
        config = super(Decoder, self).get_config()
        config.update({"output_dim": self.output_dim})
        config.update({"units": self.units})
        return config


class KullbackLeiblerDivergence(tf.keras.losses.Loss):
    """ Kullback-Leibler divergence loss. """

    def call(self, mean, logvar):
        loss = - 0.5 * (1 + logvar - tf.square(mean) - tf.exp(logvar))
        loss = tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
        return loss

    def get_config(self):
        config = super(KullbackLeiblerDivergence, self).get_config()
        return config


class MeanSquaredError(tf.keras.losses.Loss):
    """ Mean squared error loss. """

    def call(self, inputs, outputs):
        loss = tf.keras.losses.mean_squared_error(inputs, outputs)
        loss = tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
        return loss

    def get_config(self):
        config = super(MeanSquaredError, self).get_config()
        return config


class VariationalAutoencoder(tf.keras.Model):
    """ Variational autoencoder model. """

    def __init__(
            self,
            input_dim,
            latent_dim,
            units,
            beta=1.,
            **kwargs
    ):
        super(VariationalAutoencoder, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder_units = units
        self.decoder_units = units[::-1]
        self.beta = beta * self.latent_dim / self.input_dim

        self.encoder = Encoder(latent_dim=self.latent_dim, units=self.encoder_units)
        self.decoder = Decoder(output_dim=self.input_dim, units=self.decoder_units)
        self.sampling = Sampling()

        self.kl_loss = KullbackLeiblerDivergence()
        self.mse_loss = MeanSquaredError()

        self.loss_metrics = tf.keras.metrics.Mean('loss', dtype=tf.float32)
        self.kl_metrics = tf.keras.metrics.Mean('kl', dtype=tf.float32)
        self.mse_metrics = tf.keras.metrics.Mean('mse', dtype=tf.float32)

    def call(self, inputs, **kwargs):

        if isinstance(inputs, tuple):
            inputs, _ = inputs

        mean, logvar = self.encoder(inputs)
        sample = self.sampling([mean, logvar])
        outputs = self.decoder(sample)
        return outputs

    @property
    def metrics(self):
        return [
            self.loss_metrics,
            self.kl_metrics,
            self.mse_metrics,
        ]

    def train_step(self, data):
        with tf.GradientTape() as g:
            loss = self.compute_loss(data)
        gradients = g.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {
            'loss': self.loss_metrics.result(),
            'kl': self.kl_metrics.result(),
            'mse': self.mse_metrics.result()
        }

    def get_config(self):
        config = super(VariationalAutoencoder, self).get_config()
        config.update({"beta": self.beta})
        return config

    @tf.function
    def compute_loss(self, inputs):

        if isinstance(inputs, tuple):
            inputs, _ = inputs

        mean, logvar = self.encoder(inputs)
        sample = self.sampling([mean, logvar])
        outputs = self.decoder(sample)
        kl_loss = self.kl_loss(mean, logvar)
        mse_loss = self.mse_loss(inputs, outputs)
        loss = self.beta * kl_loss + mse_loss
        self.loss_metrics.update_state(loss)
        self.kl_metrics.update_state(kl_loss)
        self.mse_metrics.update_state(mse_loss)
        return loss

    @tf.function
    def encode(self, inputs):
        mean, logvar = self.encoder(inputs)
        return mean, logvar

    @tf.function
    def decode(self, inputs):
        outputs = self.decoder(inputs)
        return outputs


