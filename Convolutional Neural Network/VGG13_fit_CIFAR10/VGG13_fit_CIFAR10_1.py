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
    VGG_13.compile( optimizer='adam',
                    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
    VGG_13.fit(train_db,epochs=50,validation_data=test_db)

if __name__=='__main__':
    train_db,test_db=get_data()
    train(train_db,test_db)


