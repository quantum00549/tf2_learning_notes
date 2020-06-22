"""
ResNet18 on CIFAR10
"""

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.datasets as datasets
from tensorflow.keras import Sequential, layers, losses, optimizers, metrics


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def preprocess(x, y):
    # [0~1]
    x = 2*tf.cast(x, dtype=tf.float32) / 255.-1
    y = tf.squeeze(y, axis=1)
    y = tf.cast(y, dtype=tf.int32)
    return x,y
(x,y), (x_test, y_test) = datasets.cifar10.load_data()
train_db = tf.data.Dataset.from_tensor_slices((x,y)).shuffle(10000).batch(256).map(preprocess)
test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(256).map(preprocess)

class BasicBlock(layers.Layer):
    # 残差模块
    # Page 185，自定义网络层
    def __init__(self,filter_num,stride=1):
        super(BasicBlock,self).__init__()
        self.conv1 = layers.Conv2D(filter_num,(3,3),strides=stride,padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')

        self.conv2 = layers.Conv2D(filter_num,(3,3),strides=1,padding='same')
        self.bn2 = layers.BatchNormalization()

        if stride != 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num,(1,1),strides=stride))
        else:
            self.downsample = lambda x:x
            # 教材中的代码，不包含x通道缺失补零的操作，应该是利用了tf的自动转换

    def call(self,inputs,training=None):
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        identity = self.downsample(inputs)
        output = layers.add([out, identity])
        output = tf.nn.relu(output)
        return output


class ResNet(keras.Model):
    def __init__(self,layer_dims,num_classes=10):
        super(ResNet, self).__init__()
        self.stem = Sequential([
            layers.Conv2D(64,(3,3),strides=(1,1)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D(pool_size=(2,2),strides=(1,1),padding='same')
        ])
        self.layer1 = self.build_resblock(64,  layer_dims[0])
        self.layer2 = self.build_resblock(128, layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)

        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes)

    def build_resblock(self, filter_num, blocks, stride=1):
        # 辅助函数
        res_blocks = Sequential()
        res_blocks.add(BasicBlock(filter_num,stride))
        # 只有第一个BasicBlock 的步长可能不为1，实现下采样
        for _ in range(1,blocks):
            res_blocks.add(BasicBlock(filter_num))
        return res_blocks

    def call(self,inputs,training=None):
        x = self.stem(inputs)
        x = self.layer4(
                self.layer3(
                    self.layer2(
                        self.layer1(x))))
        x = self.fc(self.avgpool(x))
        return x

res = ResNet([2,2,2,2])
criterion = losses.CategoricalCrossentropy(from_logits=True)
optimizer = optimizers.Adam(learning_rate=1e-3)
loss_meter = metrics.Mean()
acc_meter = metrics.Accuracy()

for epoch in range(10):
    for step, (x,y) in enumerate(train_db):
        with tf.GradientTape() as tape:
            out = res(x)
            y_onehot = tf.one_hot(y,depth=10)
            loss = criterion(y_onehot, out)
        grads = tape.gradient(loss,res.trainable_variables)
        optimizer.apply_gradients(zip(grads,res.trainable_variables))
        loss_meter.update_state(loss.numpy())
        pred = tf.argmax(out, axis=1)
        acc_meter.update_state(y,pred)
        if (step+1)%10 == 0:
            print(f'Epoch:{epoch+1},Step:{step+1},Loss:{loss_meter.result().numpy()},Accuracy:{acc_meter.result().numpy()}')
            loss_meter.reset_states()
            acc_meter.reset_states()
res.save_weights('ResNet18_on_CIFAR10.ckpt')
del res


res = ResNet([2,2,2,2])
res.load_weights('ResNet18_on_CIFAR10.ckpt')
accuracy = []
for _, (x,y) in enumerate(test_db):
    out = res(x)
    pred = tf.cast(tf.argmax(out, axis=1),dtype=tf.int32)
    acc = tf.reduce_mean((tf.cast(tf.equal(y, pred),dtype=tf.float32)))
    accuracy.append(acc)
accuracy = tf.reduce_mean(accuracy)
print(accuracy.numpy())
