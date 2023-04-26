import tensorflow as tf
#参数初始化很重要
model = tf.keras.Sequential(tf.keras.layers.Dense(1,use_bias=False,kernel_initializer=tf.keras.initializers.Constant(value=0.1)))
optimizer = tf.keras.optimizers.RMSprop(0.001)
model.compile(optimizer=optimizer,loss=tf.keras.losses.mae)

model.fit(x=tf.convert_to_tensor([[90,80,70],[98,95,87]],dtype=float),y=tf.convert_to_tensor([[85],[96]],dtype=float),epochs=5000)
print(model.get_weights())
y=model.predict(tf.convert_to_tensor([[90,80,70],[98,95,87]]))
print(y)