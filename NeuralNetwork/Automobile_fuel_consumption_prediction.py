#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

def get_data():
    # Download the Car Efficiency Dataset Online
    dataset_path=tf.keras.utils.get_file("auto-mpg.data","http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
    #print (dataset_path)

    #Read datasets with pandas
    column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight','Acceleration', 'Model Year', 'Origin']
    raw_dataset = pd.read_csv(dataset_path, names=column_names,na_values = "?", comment='\t',sep=" ", skipinitialspace=True)
    dataset = raw_dataset.copy()
    dataset = dataset.dropna() # delete rows with blank data

    origin = dataset.pop('Origin')
    dataset['USA'] = (origin == 1)*1.0
    dataset['Europe'] = (origin == 2)*1.0
    dataset['Japan'] = (origin == 3)*1.0

    train_dataset = dataset.sample(frac=0.8,random_state=0) #Randomly select 80% of the data as the training set
    train_labels = train_dataset.pop('MPG')
    return train_dataset,train_labels

def pre_process(train_dataset,train_labels):
    train_stats = train_dataset.describe().transpose()
    normed_train_data = (train_dataset- train_stats['mean']) / train_stats['std']
    train_db = tf.data.Dataset.from_tensor_slices((normed_train_data.values, train_labels.values))
    train_db = train_db.shuffle(100).batch(32)
    return train_db

class Network(tf.keras.Model):
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = tf.keras.layers.Dense(64, activation='relu')
        self.fc2 = tf.keras.layers.Dense(64, activation='relu')
        self.fc3 = tf.keras.layers.Dense(1)
    def call(self, inputs, training=None, mask=None):
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

def run(train_db):
    model = Network()
    model.build(input_shape=(1, 9)) #Complete the creation of internal tensors through the build function
    model.summary()
    optimizer = tf.keras.optimizers.Adam()
    losses=[]
    for epoch in range(200):
        for step, (x,y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                out = model(x)
                loss = tf.reduce_mean(tf.keras.losses.MSE(y, out))
            if step % 10 == 0:
                print(epoch, step, float(loss))
                losses.append(loss)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
    print("over")
    plt.plot(range(1, 201), losses, color='b', label='loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    train_dataset,train_labels=get_data()
    train_db=pre_process(train_dataset,train_labels)
    run(train_db)
