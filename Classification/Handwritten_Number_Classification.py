import os
import tensorflow as tf
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # avoid printing unnecessary information

(x, y), (x_val, y_val) = tf.keras.datasets.mnist.load_data()
x = 2*tf.convert_to_tensor(x, dtype=tf.float32) / 255.-1 #Convert to float tensor and scale to -1~1，which is good for training model
y = tf.convert_to_tensor(y, dtype=tf.int32)
y = tf.one_hot(y, depth=10) #Convert digital encoding to One-hot encoding
print(x.shape, y.shape)
train_dataset = tf.data.Dataset.from_tensor_slices((x, y)) #convert (x,y) to dataset
#During the calculation process, multiple pictures can be calculated at one time,we can set the number of picture(the batch).
#A dataset object with batch function can be constructed by calling the batch() function.
train_dataset = train_dataset.batch(32) #every input:[32,h,w]

model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax')])
SGD_optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)

def train_epoch(epoch):
    for step, (x, y) in enumerate(train_dataset): #Build the Gradient Recording Environment
        with tf.GradientTape() as tape:
            x=tf.reshape(x,[-1,28*28]) #flatten,[28,28] ->[1,784]
            output = model(x) #input x to model,get output.
            loss = tf.reduce_sum(tf.square(output - y)) / x.shape[0] # here we use 'mse'.
            grads = tape.gradient(loss, model.trainable_variables)#Find the gradient information of all parameters in the model 𝜕ℒ
            SGD_optimizer.apply_gradients(zip(grads, model.trainable_variables))#update parameters
    return loss

def train():
    losses=[]
    print("Start running...")
    for epoch in range(10):
        print("running {}...".format(epoch))
        losses.append(train_epoch(epoch))
    print("end.")
    plt.plot(range(1,11),losses, color='b', marker='o', label='loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.savefig('loss.png')

if __name__ == '__main__':
    train()