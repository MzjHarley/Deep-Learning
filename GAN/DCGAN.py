import tensorflow as tf
import os
import numpy as np
from scipy.misc import toimage
from dataset import make_anime_dataset
import glob


class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.Transpose_conv1 = tf.keras.layers.Conv2DTranspose(512, 4, 1, 'valid', use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.Transpose_conv2 = tf.keras.layers.Conv2DTranspose(256, 4, 2, 'same', use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.Transpose_conv3 = tf.keras.layers.Conv2DTranspose(128, 4, 2, 'same', use_bias=False)
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.Transpose_conv4 = tf.keras.layers.Conv2DTranspose(64, 4, 2, 'same', use_bias=False)
        self.bn4 = tf.keras.layers.BatchNormalization()
        self.Transpose_conv5 = tf.keras.layers.Conv2DTranspose(3, 4, 2, 'same', use_bias=False)

    def call(self, inputs, training=None):
        x = inputs
        # [b,1,1,100],input_size=1
        x = tf.nn.relu(tf.reshape(x, (x.shape[0], 1, 1, x.shape[1])))

        # [b,1,1,100]=>[b,(input_size-1)*stride+kernel_size,(input_size-1)*stride+kernel_size,512]=[b,4,4,512]
        x = tf.nn.relu(self.bn1(self.Transpose_conv1(x), training=training))

        # [b,4,4,512]=>[b,input_size*stride,input_size*stride,256]=[b,8,8,256]
        x = tf.nn.relu(self.bn2(self.Transpose_conv2(x), training=training))

        # [b,8,8,256]=>[b,input_size*stride,input_size*stride,256]=[b,16,16,128]
        x = tf.nn.relu(self.bn3(self.Transpose_conv3(x), training=training))

        # [b,16,16,128]=>[b,input_size*stride,input_size*stride,256]=[b,32,32,64]
        x = tf.nn.relu(self.bn4(self.Transpose_conv4(x), training=training))

        # [b,32,32,64]=>[b,input_size*stride,input_size*stride,256]=[b,64,64,3]
        x = tf.tanh(self.Transpose_conv5(x))
        return x


class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(64, 4, 2, 'valid', use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(128, 4, 2, 'valid', use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(256, 4, 2, 'valid', use_bias=False)
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.conv4 = tf.keras.layers.Conv2D(512, 3, 1, 'valid', use_bias=False)
        self.bn4 = tf.keras.layers.BatchNormalization()
        self.conv5 = tf.keras.layers.Conv2D(1024, 3, 1, 'valid', use_bias=False)
        self.bn5 = tf.keras.layers.BatchNormalization()
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(1)

    def call(self, inputs, training=None):
        # [b,64,64,3]=>[b,31,31,64]
        x = tf.nn.leaky_relu(self.bn1(self.conv1(inputs), training=training))

        # [b,31,31,64]=>[b,14,14,128]
        x = tf.nn.leaky_relu(self.bn2(self.conv2(x), training=training))

        # [b,14,14,128]=>[b,6,6,256]
        x = tf.nn.leaky_relu(self.bn3(self.conv3(x), training=training))

        # [b,6,6,256]=>[b,4,4,512]
        x = tf.nn.leaky_relu(self.bn4(self.conv4(x), training=training))

        # [b,4,4,512]=>[b,2,2,1024]
        x = tf.nn.leaky_relu(self.bn5(self.conv5(x), training=training))

        # [b,2,2,1024]=>[b,1,1024]
        x = self.pool(x)

        # [b,1,1024]=>[b,1024]
        x = self.flatten(x)

        # [b,1024]=>[b,1.]
        logits = self.fc(x)
        return logits


def d_loss_fn(generator, discriminator, batch_z, batch_x, is_training):
    fake_image = generator(batch_z, is_training)
    d_fake_logits = discriminator(fake_image, is_training)
    d_real_logits = discriminator(batch_x, is_training)
    loss = celoss_ones(d_real_logits) + celoss_zeros(d_fake_logits)
    return loss


def celoss_ones(logits):
    y = tf.ones_like(logits)
    loss = tf.keras.losses.binary_crossentropy(y, logits, from_logits=True) #here，from_logits=True :sigmoid(logits)
    return tf.reduce_mean(loss)


def celoss_zeros(logits):
    y = tf.zeros_like(logits)
    loss = tf.keras.losses.binary_crossentropy(y, logits, from_logits=True)#here，from_logits=True :sigmoid(logits)
    return tf.reduce_mean(loss)


def g_loss_fn(generator, discriminator, batch_z, is_training):
    fake_image = generator(batch_z, is_training)
    d_fake_logits = discriminator(fake_image, is_training)
    loss = celoss_ones(d_fake_logits)
    return loss


def get_data():
    img_path = glob.glob(r'.\images\*.jpg') #’r‘ make string unescape
    dataset, img_shape, _ = make_anime_dataset(img_path, 64, resize=64)
    # img_shape:[64,64,3],image's every point:[-1,1]
    dataset = dataset.repeat(100)
    db_iter = iter(dataset)
    # db_iter's every element:[64,64,64,3]
    return db_iter


def train():
    generator = Generator()
    generator.build(input_shape=(4, 100))
    generator.summary()
    discriminator = Discriminator()
    discriminator.build(input_shape=(4, 64, 64, 3))
    discriminator.summary()

    g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002,beta_1=0.5)
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002,beta_1=0.5)

    d_losses, g_losses = [], []

    db_iter = get_data()

    for epoch in range(3000000):
        for i in range(5):
            batch_z = tf.random.normal([64, 100])
            batch_x = next(db_iter)  # [64,64,64,3]

            with tf.GradientTape() as tape:
                d_loss = d_loss_fn(generator, discriminator, batch_z, batch_x, is_training=True)
            grads = tape.gradient(d_loss, discriminator.trainable_variables)
            d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

        batch_z = tf.random.normal([64, 100])
        with tf.GradientTape() as tape:
            g_loss = g_loss_fn(generator, discriminator, batch_z, is_training=True)
        grads = tape.gradient(g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

        if epoch % 100 == 0:
            print(epoch, 'd-loss:', np.float(d_loss.numpy()), 'g-loss:', np.float(g_loss.numpy()))
            z = tf.random.normal([100, 100])
            fake_image = generator(z, training=False) #[100,64,64,3]
            img_path = os.path.join('gan_images', 'gan-%d.png' % epoch)
            save_result(fake_image.numpy(), 10, img_path)

            d_losses.append(float(d_loss))
            g_losses.append(float(g_loss))
            if epoch % 10000 == 1:
                print(d_losses)
                print(g_losses)
                generator.save_weights('generator.ckpt')
                discriminator.save_weights('discriminator.ckpt')


def save_result(val_out, val_block_size, image_path):
    def preprocess(img):
        img = ((img + 1.0) * 127.5).astype(np.uint8)  # img [-1,1]->[0,255]
        return img

    preprocesed = preprocess(val_out) #[100,64,64,3]
    final_image = np.array([])
    single_row = np.array([])

    for b in range(val_out.shape[0]):
        if single_row.size == 0:
            single_row = preprocesed[b, :, :, :] #[64,64,3]
        else:
            single_row = np.concatenate((single_row, preprocesed[b, :, :, :]), axis=1)

        if (b + 1) % val_block_size == 0:
            if final_image.size == 0:
                #single_row.shape:#[64,640,3]
                final_image = single_row
            else:
                final_image = np.concatenate((final_image, single_row), axis=0)
            single_row = np.array([])
    #final_image:[640,640,3]
    if final_image.shape[2] == 1:
        final_image = np.squeeze(final_image, axis=2)
    toimage(final_image).save(image_path)


if __name__ == '__main__':
    train()