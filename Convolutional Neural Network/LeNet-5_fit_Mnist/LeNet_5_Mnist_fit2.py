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
        #tf.keras.layers.ReLU(), #here, after testing, when we don't use it,the accuracy is better.
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
    LeNet.compile(optimizer='adam',loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
    LeNet.fit(train_db,validation_data=val_db,epochs=30)

if __name__=='__main__':
    train()