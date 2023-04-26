import tensorflow as tf
#参数初始化很重要
model = tf.keras.layers.Dense(1,use_bias=False,kernel_initializer=tf.keras.initializers.Constant(value=0.1))
optimizer = tf.keras.optimizers.RMSprop(0.001)
for i in range(5000):
    with tf.GradientTape() as tape:
        y5 = model(tf.convert_to_tensor([[90,80,70],[98,95,87]],dtype=float))
        loss = tf.math.reduce_mean(tf.abs(y5-tf.convert_to_tensor([[85],[96]],dtype=float)))
    test=tape.gradient(loss,model.trainable_weights)
    optimizer.apply_gradients(zip(test,model.trainable_weights))
    print(model.trainable_weights)

y5 = model(tf.convert_to_tensor([[90,80,70],[98,95,87]]))
print(y5)
