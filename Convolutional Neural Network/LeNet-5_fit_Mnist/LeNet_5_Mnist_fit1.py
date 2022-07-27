import tensorflow as tf

def pre_process(x,y):
    x=tf.cast(x,dtype=tf.float32)
    x = tf.expand_dims(x, axis=2)
    y=tf.one_hot(tf.cast(y,dtype=tf.int32),depth=10)
    return x,y

def get_data():
    (x,y),(x_val,y_val)= tf.keras.datasets.mnist.load_data()
    train_db=tf.data.Dataset.from_tensor_slices((x,y)).map(pre_process).batch(128)
    val_db=tf.data.Dataset.from_tensor_slices((x_val,y_val)).map(pre_process).batch(128)
    return train_db,val_db

def LeNet_5():
    model=tf.keras.Sequential([
        tf.keras.layers.Conv2D(6,kernel_size=3,strides=1),
        tf.keras.layers.MaxPool2D(pool_size=2,strides=2),
        #tf.keras.layers.ReLU(), #here ,you decide by yourself to add or not.
        tf.keras.layers.Conv2D(16, kernel_size=3, strides=1),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120,activation='relu'),
        tf.keras.layers.Dense(84, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu')
        ])
    model.build(input_shape=(4,28,28,1))
    model.summary()
    return model

def train():
    train_db, val_db = get_data()
    LeNet=LeNet_5()
    optimizer=tf.keras.optimizers.Adam(0.001)
    for i in range(30):
        for step,(x,y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                out = LeNet(x)
                loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y,out,from_logits=True))
            grads=tape.gradient(loss,LeNet.trainable_variables)
            optimizer.apply_gradients(zip(grads,LeNet.trainable_variables))

        correct,total=0,0
        for x,y in val_db:
            out=LeNet(x)
            correct +=tf.cast(tf.reduce_sum(tf.cast(tf.equal(tf.argmax(out,axis=-1),tf.argmax(y,axis=-1)),dtype=tf.int32)),dtype=tf.float32)
            total+=x.shape[0]
        print("val Acc:",(correct/total).numpy())

if __name__=='__main__':
    train()