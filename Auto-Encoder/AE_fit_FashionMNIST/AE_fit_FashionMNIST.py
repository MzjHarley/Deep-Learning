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


class AE(tf.keras.Model):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = tf.keras.Sequential([tf.keras.layers.Dense(256, activation='relu'),
                                            tf.keras.layers.Dense(128, activation='relu'),
                                            tf.keras.layers.Dense(20)])
        self.decoder = tf.keras.Sequential([tf.keras.layers.Dense(128, activation='relu'),
                                            tf.keras.layers.Dense(256, activation='relu'),
                                            tf.keras.layers.Dense(784)])

    def call(self, inputs, training=None):
        z = self.encoder(inputs)
        prob = self.decoder(z)
        return prob


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
    model = AE()
    model.build(input_shape=(None, 784))
    model.summary()
    train_db, test_db = get_data()
    optimizer = tf.keras.optimizers.Adam()
    for epoch in range(100):
        for step, x in enumerate(train_db):
            with tf.GradientTape() as tape:
                xt = model(x)
                loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=xt))
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            if step % 100 == 0:
                print(epoch, step, float(loss))
        if epoch % 10 == 0:
            x = next(iter(test_db)) #[128,784]
            xt = tf.reshape(tf.sigmoid(model(x)), [-1, 28, 28]) #[128,28,28]
            x_concat = tf.cast(tf.concat([tf.reshape(x, [-1, 28, 28])[:50], xt[:50]], axis=0) * 255.,dtype=tf.uint8)  # [100,28,28]
            save_images(x_concat, f'AE_images/epoch_{epoch}.png')


if __name__ == '__main__':
    main()
