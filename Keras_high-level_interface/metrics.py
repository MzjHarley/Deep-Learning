import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_data():
    (x, y), (x_val, y_val) = tf.keras.datasets.mnist.load_data()
    print(x.shape, y.shape, x_val.shape, y_val.shape)
    x = tf.reshape(2 * tf.cast(x, dtype=tf.float32) / 255. - 1, [-1, 28 * 28])
    y = tf.one_hot(tf.cast(y, dtype=tf.int32), depth=10)
    x_val = tf.reshape(2 * tf.cast(x_val, dtype=tf.float32) / 255. - 1, [-1, 28 * 28])
    y_val = tf.one_hot(tf.cast(y_val, dtype=tf.int32), depth=10)
    print(x.shape, y.shape, x_val.shape, y_val.shape)
    train_db = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(60000).batch(128).repeat(10)
    val_db = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(128)
    return train_db,val_db

def get_network():
    network = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10)])
    network.build(input_shape=(None, 28 * 28))
    network.summary()
    return network

def train(network,train_db,val_db):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    acc_meter = tf.keras.metrics.Accuracy()
    loss_meter = tf.keras.metrics.Mean()

    for step, (x, y) in enumerate(train_db):
        with tf.GradientTape() as tape:
            out = network(x)
            loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y, out, from_logits=True))
            loss_meter.update_state(loss)
        grads = tape.gradient(loss, network.trainable_variables)
        optimizer.apply_gradients(zip(grads, network.trainable_variables))
        if step % 100 == 0:
            print(step, 'loss:', loss_meter.result().numpy())
            loss_meter.reset_states()

        if step % 500 == 0:
            total, total_correct = 0, 0
            acc_meter.reset_states()
            for step, (x, y) in enumerate(val_db):
                out = network(x)
                correct = tf.equal(tf.argmax(out, axis=1), tf.argmax(y, axis=1))
                total_correct += tf.reduce_sum(tf.cast(correct, dtype=tf.int32)).numpy()
                total += x.shape[0]
                acc_meter.update_state(tf.argmax(y, axis=1), tf.argmax(out, axis=1))
            print('Evaluate Acc:', total_correct / total, acc_meter.result().numpy())

if __name__=='__main__':
    train_db, val_db=get_data()
    network=get_network()
    train(network,train_db,val_db)