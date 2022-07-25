import tensorflow as tf

def get_data():
    (x,y),(x_val,y_val)=tf.keras.datasets.mnist.load_data()
    print(x.shape,y.shape,x_val.shape,y_val.shape)

    x = tf.reshape(2*tf.convert_to_tensor(x,dtype=tf.float32)/255.-1,[-1,28*28])
    y=tf.one_hot(tf.convert_to_tensor(y,dtype=tf.int32),depth=10)
    x,x_test=tf.split(x,num_or_size_splits=[50000,10000]) # !!!
    y,y_test = tf.split(y, num_or_size_splits=[50000, 10000])# !!!
    x_val = tf.reshape(2 * tf.convert_to_tensor(x_val, dtype=tf.float32) / 255. - 1, [-1, 28 * 28])
    y_val = tf.one_hot(tf.convert_to_tensor(y_val, dtype=tf.int32), depth=10)

    train_db= tf.data.Dataset.from_tensor_slices((x,y)).batch(128)
    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(128)# !!!
    val_db = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(128)
    return x,y,train_db,val_db,test_db

def train(x,y,train_db,val_db,test_db):
    model=tf.keras.Sequential([tf.keras.layers.Dense(256,activation='relu'),
                               tf.keras.layers.Dense(128,activation='relu'),
                               tf.keras.layers.Dense(64,activation='relu'),
                               tf.keras.layers.Dense(32,activation='relu'),
                               tf.keras.layers.Dense(10)])
    model.build(input_shape=(None,28*28))
    model.summary()
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.fit(train_db,epochs=5,validation_data=val_db,validation_freq=2)
    model.evaluate(test_db)# !!!

    out=tf.argmax(model.predict(x),axis=1)
    y=tf.argmax(y,axis=1)
    print(out[:10])
    print(y[:10])



if __name__=='__main__':
    x,y,train_db,val_db,test_db=get_data()
    train(x,y,train_db,val_db,test_db)