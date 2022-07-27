import  tensorflow as tf

class BasicBlock(tf.keras.layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.Activation('relu')
        self.conv2 = tf.keras.layers.Conv2D(filter_num, (3, 3), strides=1, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()

        if stride != 1:
            self.downsample = tf.keras.Sequential([tf.keras.layers.Conv2D(filter_num, (1, 1), strides=stride)])
        else:
            self.downsample = lambda x:x

    def call(self, inputs, training=None):
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        identity = self.downsample(inputs)
        output = tf.keras.layers.add([out, identity])
        output = tf.nn.relu(output)
        return output


class ResNet(tf.keras.Model):
    def __init__(self, layer_dims, num_classes=10):
        super(ResNet, self).__init__()
        self.stem = tf.keras.Sequential([tf.keras.layers.Conv2D(64, (3, 3), strides=1),
                                tf.keras.layers.BatchNormalization(),
                                tf.keras.layers.Activation('relu'),
                                tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=1, padding='same')
                                ])
        self.layer1 = self.build_resblock(64,  layer_dims[0])
        self.layer2 = self.build_resblock(128, layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)
        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(num_classes)

    def call(self, inputs, training=None):
        x = self.stem(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x

    def build_resblock(self, filter_num, blocks, stride=1):
        res_blocks = tf.keras.Sequential()
        res_blocks.add(BasicBlock(filter_num, stride))
        for _ in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, stride=1))
        return res_blocks

def ResNet18():
    return ResNet([2, 2, 2, 2])

#def ResNet34():
    #return ResNet([3, 4, 6, 3])

def preprocess(x, y):
    x = 2*tf.cast(x, dtype=tf.float32) / 255. - 1
    y = tf.squeeze(y, axis=-1)
    y = tf.one_hot(tf.cast(y, dtype=tf.int32), depth=10)
    return x,y

def get_data():
    (x,y), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    print(x.shape, y.shape, x_test.shape, y_test.shape)
    train_db = tf.data.Dataset.from_tensor_slices((x,y)).map(preprocess).batch(128)
    test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test)).map(preprocess).batch(128)
    return train_db,test_db

def train(train_db,test_db):
    model = ResNet18()
    model.build(input_shape=(None, 32, 32, 3))
    model.summary()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    for epoch in range(100):
        for step, (x,y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                out = model(x)
                loss = tf.reduce_mean(tf.losses.categorical_crossentropy(y, out, from_logits=True))
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            if step %50 == 0:
                print(epoch, step, 'loss:', float(loss))

        total_num,total_correct = 0,0
        for x,y in test_db:
            out = model(x)
            correct = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(out, axis=1), tf.argmax(y, axis=1)), dtype=tf.float32))
            total_num += x.shape[0]
            total_correct += correct
        print(epoch, 'acc:', (total_correct / total_num).numpy())

if __name__ == '__main__':
    train_db, test_db=get_data()
    train(train_db,test_db)
