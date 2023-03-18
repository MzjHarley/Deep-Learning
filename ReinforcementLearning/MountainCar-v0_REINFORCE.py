import tensorflow as tf
import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('MountainCar-v0')
env.seed(2333) # make every env.reset() same

class Policy(tf.keras.Model):
    def __init__(self):
        super(Policy, self).__init__()
        self.data = []
        self.fc1 = tf.keras.layers.Dense(128, kernel_initializer='he_normal')
        self.fc2 = tf.keras.layers.Dense(3, kernel_initializer='he_normal')
        self.optimizer = tf.keras.optimizers.Adam(lr=0.0002)

    def call(self, inputs, training=None):
        x = tf.nn.relu(self.fc1(inputs))
        x = tf.nn.softmax(self.fc2(x), axis=1)
        return x

    def put_data(self, item):
        self.data.append(item)

    def train_net(self, tape):
        R = 0
        for r, log_prob in self.data[::-1]:
            R = r + 0.98 * R
            loss=-log_prob*R
            with tape.stop_recording():
                grads = tape.gradient(loss, self.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.data = []

def main():
    model = Policy()
    model.build(input_shape=(4,2))
    model.summary()

    score = 0.0
    score_list = []
    for epoch in range(400):
        s = env.reset()
        with tf.GradientTape(persistent=True) as tape:
            for _ in range(501):
                s = tf.expand_dims(tf.constant(s, dtype=tf.float32), axis=0)
                prob = model(s)
                b=tf.random.categorical(tf.math.log(prob), 1) # [1,1]
                # tf.random.categorical:return index,shape[b,num_samples],the greater the probability, the easier it is to be sampled.
                a = int(b[0])  # Tensor,shape=(1,) -> int
                s_next, reward, end, diagnose_info = env.step(a)
                model.put_data((reward, tf.math.log(prob[0][a])))
                s = s_next
                score += reward
                if epoch > 0:
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
    plt.savefig('balance_bar_game.png')


if __name__ == '__main__':
    main()