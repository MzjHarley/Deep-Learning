import tensorflow as tf
import numpy as np
from PIL import Image


def preprocess(x):
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [28 * 28])
    return x


def get_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    # print(x_train.shape,x_test.shape)#(60000, 28, 28) (10000, 28, 28)
    train_db = tf.data.Dataset.from_tensor_slices(x_train).map(preprocess).shuffle(128 * 5).batch(128)
    test_db = tf.data.Dataset.from_tensor_slices(x_test).map(preprocess).batch(128)
    return train_db, test_db


class VAE(tf.keras.Model):
    def __init__(self):
        super(VAE, self).__init__()
        # Encoder
        self.fc1 = tf.keras.layers.Dense(128, activation='relu')
        self.fc2 = tf.keras.layers.Dense(20)
        self.fc3 = tf.keras.layers.Dense(20)
        # Decoder
        self.fc4 = tf.keras.layers.Dense(128, activation='relu')
        self.fc5 = tf.keras.layers.Dense(784)

    def encoder(self, x):
        x = self.fc1(x)
        mean = self.fc2(x)
        log_var = self.fc3(x)
        return mean, log_var

    def decoder(self, z):
        out = self.fc4(z)
        out = self.fc5(out)
        return out

    def reparameterize(self, mean, log_var):
        eps = tf.random.normal(log_var.shape)
        std = tf.exp(log_var)
        z = mean + std * eps
        return z

    def call(self, inputs, training=None):
        mean, log_var = self.encoder(inputs)
        z = self.reparameterize(mean, log_var)
        out = self.decoder(z)
        return out, mean, log_var


def save_images(imgs, name):
    new_image = Image.new('L', (280, 280))
    index = 0
    for i in range(0, 280, 28):
        for j in range(0, 280, 28):
            temp_image = imgs[index]  # [28,28]
            temp_image = Image.fromarray(np.asarray(temp_image), mode='L')
            new_image.paste(temp_image, (i, j))
            index += 1
    new_image.save(name)


def main():
    model = VAE()
    model.build(input_shape=(1, 784))
    model.summary()
    train_db, test_db = get_data()
    optimizer = tf.keras.optimizers.Adam()
    for epoch in range(100):
        for step, x in enumerate(train_db):
            with tf.GradientTape() as tape:
                xt, mean, log_var = model(x)
                recon_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=xt))
                KL_loss = tf.reduce_mean(-0.5 * (1 - mean ** 2 - tf.exp(log_var) ** 2) - log_var)
                loss = recon_loss + KL_loss

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            if step % 100 == 0:
                print(epoch, step, float(loss))
        if epoch % 10 == 0:
            # only use decoder to generate picture
            z = tf.random.normal((128, 20))
            xt = tf.sigmoid(model.decoder(z))
            xt = tf.cast(tf.reshape(xt, [-1, 28, 28]).numpy() * 255.,dtype=tf.uint8)
            save_images(xt, 'VAE_images/epoch_%d_sampled.png' % epoch)
            # use Model to generate picture
            x = next(iter(test_db))  # [128,784]
            xt, _, _ = model(x)
            xt = tf.reshape(xt, [-1, 28, 28])  # [128,28,28]
            x_concat = tf.cast(tf.concat([tf.reshape(x, [-1, 28, 28])[:50], xt[:50]], axis=0) * 255.,
                               dtype=tf.uint8)  # [100,28,28]
            save_images(x_concat, f'VAE_images/epoch_{epoch}.png')


if __name__ == '__main__':
    main()
