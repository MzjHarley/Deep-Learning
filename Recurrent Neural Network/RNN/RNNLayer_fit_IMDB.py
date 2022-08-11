import tensorflow as tf


def get_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=10000)
    # print(x_train.shape,y_train.shape,x_test.shape,y_test.shape) #(25000,) (25000,) (25000,) (25000,)
    # print(len(x_train[0]),len(x_train[1]),len(x_test[0]),len(x_test[1])) #218 189 68 260
    #print(y_train) #eg.[1,0,...,1]
    #print(y_test) #eg.[0,1,...,0]
    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=80)
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=80)
    # print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)#(25000, 80) (25000,) (25000, 80) (25000,)
    db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(128, drop_remainder=True)
    db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(128, drop_remainder=True)
    return db_train, db_test


class MyRNN(tf.keras.Model):
    def __init__(self, units):
        super(MyRNN, self).__init__()
        self.embedding = tf.keras.layers.Embedding(10000, 100, input_length=80)  # [128, 80, 100]
        self.rnn = tf.keras.Sequential([
            tf.keras.layers.SimpleRNN(units, dropout=0.5, return_sequences=True),
            tf.keras.layers.SimpleRNN(units, dropout=0.5)
        ])
        self.outlayer = tf.keras.Sequential([
            tf.keras.layers.Dense(32),
            tf.keras.layers.Dropout(rate=0.5),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(1,activation='sigmoid')])

    def call(self, inputs, training=None):
        x = inputs  # [128,80]
        x = self.embedding(x)  # [128,80,100]
        # [b,100]->[b,64]
        x = self.rnn(x,training)
        # [b,64]->[b,1]
        prob = self.outlayer(x,training)
        return prob


def main():
    db_train, db_test= get_data()
    model = MyRNN(64)
    model.build(input_shape=(None, 80))
    model.summary()
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    model.fit(db_train, epochs=20, validation_data=db_test)

if __name__ == '__main__':
    main()
