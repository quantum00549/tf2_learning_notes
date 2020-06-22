import  tensorflow as tf
from    tensorflow.keras import datasets, layers, optimizers, Sequential, metrics, losses


def preprocess(x, y):
    """
    x is a simple image, not a batch
    """
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [28*28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x,y


batchsz = 128
(x, y), (x_val, y_val) = datasets.mnist.load_data()
print('datasets:', x.shape, y.shape, x.min(), x.max())



db = tf.data.Dataset.from_tensor_slices((x,y))
db = db.map(preprocess).shuffle(60000).batch(batchsz)
ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
ds_val = ds_val.map(preprocess).batch(batchsz)



net = Sequential([layers.Dense(256,activation='relu'),
                  layers.Dense(128,activation='relu'),
                  layers.Dense(64,activation='relu'),
                  layers.Dense(32,activation='relu'),
                  layers.Dense(10)])
net.compile(optimizer=optimizers.Adam(lr=1e-2),
            loss = losses.CategoricalCrossentropy(from_logits=True),
            metrics = ['accuracy'])
history = net.fit(db,epochs=2,validation_data=ds_val,validation_freq=2)

