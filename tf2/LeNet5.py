"""
LeNet5笔记
模型保存见Page 182
测量容器见Page 190
模型介绍见Page 240
"""


import tensorflow as tf
import tensorflow.keras.datasets as datasets
from tensorflow.keras import Sequential, layers, losses, optimizers, metrics

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

(x, y), _ = datasets.mnist.load_data()
train_db = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(10000).batch(128)

net = Sequential([
    layers.Conv2D(6, kernel_size=3, strides=1),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=2, strides=2),
    layers.ReLU(),
    layers.Conv2D(16, kernel_size=3, strides=1),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=2, strides=2),
    layers.ReLU(),
    layers.Flatten(),

    layers.Dense(120, activation='relu'),
    layers.Dense(84, activation='relu'),
    layers.Dense(10)
])
net.build(input_shape=(None, 28, 28, 1))

criterion = losses.CategoricalCrossentropy(from_logits=True)
optimizer = optimizers.Adam(learning_rate=1e-3)
loss_meter = metrics.Mean()
acc_meter = metrics.Accuracy()

for epoch in range(10):
    for step, (x, y) in enumerate(train_db):
        with tf.GradientTape() as tape:
            x = tf.expand_dims(x, axis=3)
            out = net(tf.cast(x, dtype=tf.float32), training=True)
            y_onehot = tf.one_hot(
                tf.cast(y, dtype=tf.int32),depth=10
            )
            loss = criterion(y_onehot, out)
        grads = tape.gradient(loss, net.trainable_variables)
        optimizer.apply_gradients(zip(grads, net.trainable_variables))
        loss_meter.update_state(float(loss))
        pred = tf.cast(
            tf.argmax(out, axis=1),dtype=tf.int32
        )
        acc_meter.update_state(y, pred)
        if (step + 1) % 10 == 0:
            print(epoch, step+1, loss_meter.result().numpy(), acc_meter.result().numpy())
            loss_meter.reset_states()
            acc_meter.reset_states()
net.save('LeNet5.h5')
del net