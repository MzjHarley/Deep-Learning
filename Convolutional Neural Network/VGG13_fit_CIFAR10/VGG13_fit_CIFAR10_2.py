import tensorflow as tf

def pre_process(x,y):
    x=2*tf.cast(x,dtype=tf.float32)/255.-1
    y=tf.squeeze(y,axis=-1)
    y=tf.one_hot(tf.cast(y,dtype=tf.int32),depth=10)
    return x,y

def get_data():
    (x,y),(x_test,y_test) = tf.keras.datasets.cifar10.load_data()
    #print(x.shape,y.shape)
    train_db=tf.data.Dataset.from_tensor_slices((x,y)).map(pre_process).shuffle(1000).batch(128)
    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).map(pre_process).shuffle(1000).batch(128)
    return train_db,test_db

#train_db,test_db=get_data()
#sameple=next(iter(train_db))
#print(sameple[0].shape,sameple[1].shape)

def VGG13():
    # 5 conv-conv-pooling and 3 fc
    model=tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation=tf.nn.relu),
        tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation=tf.nn.relu),
        tf.keras.layers.MaxPool2D(pool_size=[2,2],strides=2,padding='same'),

        tf.keras.layers.Conv2D(128, kernel_size=3, padding='same', activation=tf.nn.relu),
        tf.keras.layers.Conv2D(128, kernel_size=3, padding='same', activation=tf.nn.relu),
        tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

        tf.keras.layers.Conv2D(256, kernel_size=3, padding='same', activation=tf.nn.relu),
        tf.keras.layers.Conv2D(256, kernel_size=3, padding='same', activation=tf.nn.relu),
        tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

        tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', activation=tf.nn.relu),
        tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', activation=tf.nn.relu),
        tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

        tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', activation=tf.nn.relu),
        tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', activation=tf.nn.relu),
        tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256,activation=tf.nn.relu),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=None)
    ])
    model.build(input_shape=(4,32,32,3))
    model.summary()
    return model

def train(train_db,test_db):
    VGG_13=VGG13()
    optimizer=tf.keras.optimizers.Adam(0.001)
    for i in range(50):
        for step,(x,y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                out=VGG_13(x)
                loss=tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y,out,from_logits=True))
            grads=tape.gradient(loss,VGG_13.trainable_variables)
            optimizer.apply_gradients(zip(grads,VGG_13.trainable_variables))

        correct, total = 0, 0
        for x, y in test_db:
            out = VGG_13(x)
            correct += tf.cast(
                tf.reduce_sum(tf.cast(tf.equal(tf.argmax(out, axis=-1), tf.argmax(y, axis=-1)), dtype=tf.int32)),
                dtype=tf.float32)
            total += x.shape[0]
        print("val Acc:", (correct / total).numpy())

if __name__=='__main__':
    train_db,test_db=get_data()
    train(train_db,test_db)


