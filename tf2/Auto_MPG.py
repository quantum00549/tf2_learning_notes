import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses, metrics
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.keras.datasets as datasets
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv('auto-mpg.data', names=column_names, na_values="?", comment='\t', sep=" ",
                          skipinitialspace=True)
data = raw_dataset.copy().dropna()
origin = data.pop('Origin')
data['USA'] = (origin == 1) * 1.0
data['Europe'] = (origin == 2) * 1.0
data['Japan'] = (origin == 3) * 1.0
train = data.sample(frac=0.8, random_state=0)
test = data.drop(index=train.index)
train_labels = train.pop('MPG')
test_labels = test.pop('MPG')

scaler = StandardScaler()
normed_train, normed_test = scaler.fit_transform(train), scaler.fit_transform(test)

train_db = tf.data.Dataset.from_tensor_slices((normed_train, train_labels.values)).shuffle(100).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((normed_test, test_labels.values))


class DNN(keras.Model):
    def __init__(self):
        super(DNN, self).__init__()
        self.fc1 = layers.Dense(64, activation='relu')
        self.fc2 = layers.Dense(64, activation='relu')
        self.fc3 = layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        x = self.fc1(inputs)
        x = self.fc2(x)
        return self.fc3(x)


model = DNN()
model.build(input_shape=(None, 9))
print(model.summary())
optimizer = tf.keras.optimizers.RMSprop(1e-3)
L = []
loss_meter = metrics.Mean()
for epoch in range(20):
    loss_meter.reset_states()
    for step, (x, y) in enumerate(train_db):
        with tf.GradientTape() as tape:
            out = model(x)
            loss = tf.reduce_mean(losses.MAE(y, out))
        if (step + 1) % 10 == 0:
            print(epoch, step + 1, loss.numpy())
            L.append(loss.numpy())
            loss_meter.update_state(float(loss))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

plt.plot(L)
plt.show()
