import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.datasets as datasets

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
net = keras.models.load_model('LeNet5.h5')

_, (x_val, y_val) = datasets.mnist.load_data()
x_val = tf.expand_dims(x_val, axis=3)
out = net(
    tf.cast(x_val, dtype=tf.float32)
)
pred = tf.argmax(out, axis=1)

correct = tf.reduce_mean(tf.cast(tf.equal(pred, y_val),dtype=tf.float32))
print(correct.numpy())