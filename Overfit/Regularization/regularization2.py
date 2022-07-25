import tensorflow as tf

def get_data():
    (x,y),(x_val,y_val)=tf.keras.datasets.mnist.load_data()
    print(x.shape,y.shape,x_val.shape,y_val.shape)

    x = tf.reshape(2*tf.convert_to_tensor(x,dtype=tf.float32)/255.-1,[-1,28*28])
    y=tf.one_hot(tf.convert_to_tensor(y,dtype=tf.int32),depth=10)
    x_val = tf.reshape(2 * tf.convert_to_tensor(x_val, dtype=tf.float32) / 255. - 1, [-1, 28 * 28])
    y_val = tf.one_hot(tf.convert_to_tensor(y_val, dtype=tf.int32), depth=10)

    train_db= tf.data.Dataset.from_tensor_slices((x,y)).batch(128).repeat(10)
    val_db = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(128)
    return x,y,train_db,val_db

def train(x,y,train_db,val_db):
    model=tf.keras.Sequential([tf.keras.layers.Dense(256,activation='relu'),
                               tf.keras.layers.Dense(128,activation='relu'),
                               tf.keras.layers.Dense(64,activation='relu'),
                               tf.keras.layers.Dense(32,activation='relu'),
                               tf.keras.layers.Dense(10)])
    model.build(input_shape=(None,28*28))
    model.summary()
    optimizer=tf.keras.optimizers.Adam(0.01)

    for step, (x, y) in enumerate(train_db):
        with tf.GradientTape() as tape:
            out = model(x)
            loss = tf.reduce_mean(tf.losses.categorical_crossentropy(y, out, from_logits=True))

            # get regularization
            loss_regularization = []
            for p in model.trainable_variables:
                loss_regularization.append(tf.nn.l2_loss(p))
            loss_regularization = tf.reduce_sum(tf.stack(loss_regularization))
            loss += 0.0001 * loss_regularization
            # end

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 100 == 0:
            print(step, 'loss:', float(loss), 'loss_regularization:', float(loss_regularization))

        if step % 500 == 0:
            total, total_correct = 0, 0
            for step, (x, y) in enumerate(val_db):
                out = model(x)
                correct = tf.equal(tf.argmax(out, axis=1), tf.argmax(y, axis=1))
                total_correct += tf.reduce_sum(tf.cast(correct, dtype=tf.int32)).numpy()
                total += x.shape[0]
            print('Val Acc:', total_correct / total)

    out=tf.argmax(model.predict(x),axis=1)
    y=tf.argmax(y,axis=1)
    print(out[:10])
    print(y[:10])


if __name__=='__main__':
    x,y,train_db,val_db=get_data()
    train(x,y,train_db,val_db)