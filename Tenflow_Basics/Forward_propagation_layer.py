import matplotlib.pyplot as plt
import tensorflow as tf

plt.rcParams['font.size'] = 16
plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    (x, y), (x_val, y_val) = tf.keras.datasets.mnist.load_data()
    x = 2*tf.convert_to_tensor(x, dtype=tf.float32) / 255.-1
    y = tf.convert_to_tensor(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)

    train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
    train_dataset = train_dataset.batch(200)
    return train_dataset

def init_paramaters():

    #784->256->128->10
    # 1th layer's parameters
    w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
    b1 = tf.Variable(tf.zeros([256]))
    # 2nd layer's parameters
    w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
    b2 = tf.Variable(tf.zeros([128]))
    # 3rd layer's parameters
    w3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1))
    b3 = tf.Variable(tf.zeros([10]))
    return w1, b1, w2, b2, w3, b3


def train_epoch(train_dataset, w1, b1, w2, b2, w3, b3, lr=0.001):
    for step, (x, y) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            # [b,28,28]->[b,784]
            x=tf.reshape(x,[-1,28*28])

            # [b, 784]@[784, 256] + [256]
            h1 = x @ w1 + b1
            h1 = tf.nn.relu(h1)
            #[b, 256] => [b, 128]
            h2 = h1 @ w2 + b2
            h2 = tf.nn.relu(h2)
            # [b, 128] => [b, 10]
            output = h2 @ w3 + b3
            loss = tf.reduce_mean(tf.square(y - output))
        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])

        # gradient update => w=w-lr * grad,  b=b-lr * grad
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])
        w3.assign_sub(lr * grads[4])
        b3.assign_sub(lr * grads[5])

    return loss.numpy()

def train(epochs):
    losses = []
    train_dataset = load_data()
    w1, b1, w2, b2, w3, b3 = init_paramaters()
    for epoch in range(epochs):
        loss = train_epoch(train_dataset, w1, b1, w2, b2, w3, b3, lr=0.001)
        print('epoch:', epoch, 'loss:', loss)
        losses.append(loss)

    plt.plot(range(1,epochs+1), losses, color='blue', marker='s', label='loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()
    plt.savefig('loss.png')

if __name__=='__main__':
    train(20)
