import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

png = tf.io.read_file("lenna.png")
png = tf.image.decode_png(png, channels=3)
print(png.dtype)
pl=plt.figure(figsize=(24,6)) #创建一块画布

png1=tf.image.resize(png,[244,244])
print(png1.dtype)
png1=np.asarray(png1.numpy(),dtype='uint8')
ax1=pl.add_subplot(1,7,1)
ax1.imshow(png1)

png2 = tf.image.rot90(png, 2)
print(png2.dtype)
ax2=pl.add_subplot(1,7,2)
ax2.imshow(png2)

png3 = tf.image.flip_left_right(png)
print(png3.dtype)
ax3=pl.add_subplot(1,7,3)
ax3.imshow(png3)


png4 = tf.image.flip_up_down(png)
print(png4.dtype)
ax4=pl.add_subplot(1,7,4)
ax4.imshow(png4)

png5 = tf.image.random_crop(png1, [224,224,3])
print(png5.dtype)
ax4=pl.add_subplot(1,7,5)
ax4.imshow(png5)

noise=tf.random.normal(shape=png.shape,mean=0.0,stddev=1.0)
png6=tf.cast(png,dtype=tf.float32)+30*noise
png6=np.asarray(png6.numpy(),dtype='uint8')
ax5=pl.add_subplot(1,7,6)
ax5.imshow(png6)

png7=png.numpy()
png7[200:400,400:450,:]=0
ax6=pl.add_subplot(1,7,7)
ax6.imshow(png7)

plt.show()