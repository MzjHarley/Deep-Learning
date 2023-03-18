import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)
env.seed(222)


class Policy(tf.keras.Model):
    def __init__(self):
        super(Policy, self).__init__()
        self.data = []
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)

        self.conv1 = tf.keras.layers.Conv2D(7, (2, 2), strides=2)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.Activation('relu')

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(12)

    def call(self, inputs, training=None):
        x = self.relu1(self.bn1(self.conv1(inputs), training=training))
        x = self.avgpool(x)
        x = tf.nn.softmax(self.fc(x), axis=1)
        return x

    def put_data(self, item):
        self.data.append(item)

    def train_net(self, tape):
        R = 0
        for r, log_prob in self.data[::-1]:
            R = r + 0.98 * R
            loss = -log_prob * R
            with tape.stop_recording():
                grads = tape.gradient(loss, self.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.data = []


def preprocess(x):
    x = 2 * tf.cast(x, dtype=tf.float32) / 255. - 1
    return x


def main():
    model = Policy()
    model.build(input_shape=(4,240,256,3))
    model.summary()

    score = 0.0
    score_list = []
    for epoch in range(400):
        s = env.reset()
        with tf.GradientTape(persistent=True) as tape:
            for _ in range(501):
                s = tf.expand_dims(tf.constant(preprocess(s), dtype=tf.float32), axis=0)
                prob = model(s,training=True)
                b = tf.random.categorical(tf.math.log(prob), 1)
                a = int(b[0])
                s_next, reward, end, diagnose_info = env.step(a)
                model.put_data((reward, tf.math.log(prob[0][a])))
                s = s_next
                score += reward
                env.render()
                if end:
                    break
            model.train_net(tape)
        del tape

        if epoch % 20 == 0 and epoch != 0:
            score_list.append(score / 20)
            print(f"epoch:{epoch}, avg score : {score / 20}")
            score = 0.0

    env.close()

    plt.plot(np.arange(len(score_list)) * 20, score_list, marker='o', mfc='r', markersize=5)
    plt.xlabel('回合数')
    plt.ylabel('总回报')
    plt.savefig('SuperMarioBros-v0_REINFORCE.png')


if __name__ == '__main__':
    main()
