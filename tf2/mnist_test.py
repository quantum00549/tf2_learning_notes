import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.datasets as datasets
from tqdm import tqdm
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def process(x,y):
    x = tf.reshape(
        tf.cast(x,dtype=tf.float32)/255.,[-1,28*28]
    )
    y = tf.one_hot(
        tf.cast(y,dtype=tf.int32),depth=10
    )
    return x, y


# (x, y), (x_val, y_val) = datasets.mnist.load_data()
# train_db = tf.data.Dataset.from_tensor_slices((x,y)).shuffle(10000).batch(128).map(process)

# lr = 1e-2
# accs, losses = [], []
#
# # 784 => 512
# w1, b1 = tf.Variable(tf.random.normal([784, 256], stddev=0.1)), tf.Variable(tf.zeros([256]))
# # 512 => 256
# w2, b2 = tf.Variable(tf.random.normal([256, 128], stddev=0.1)), tf.Variable(tf.zeros([128]))
# # 256 => 10
# w3, b3 = tf.Variable(tf.random.normal([128, 10], stddev=0.1)), tf.Variable(tf.zeros([10]))
#
# for epoch in tqdm(range(20)):
#     for step,(x,y) in tqdm(enumerate(train_db)):
#         with tf.GradientTape() as tape:
#             h1 = x @ w1 + b1
#             h1 = tf.nn.relu(h1)
#             h2 = h1 @ w2 + b2
#             h2 = tf.nn.relu(h2)
#             out = h2 @ w3 + b3
#             loss = tf.reduce_mean(tf.square(y - out))
#
#         grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
#         for w,g in zip([w1, b1, w2, b2, w3, b3], grads):
#             w.assign_sub(lr*g)
#         if (step+1)%80 == 0:
#             losses.append(loss)
#
#             h1 = x @ w1 + b1
#             h1 = tf.nn.relu(h1)
#             h2 = h1 @ w2 + b2
#             h2 = tf.nn.relu(h2)
#             out = h2 @ w3 + b3
#
#             pred = tf.argmax(out, axis=1)
#             label = tf.argmax(y, axis=1)
#             correct = tf.reduce_mean(tf.cast(tf.equal(pred,label),dtype=tf.float32))
#             accs.append(correct)
# plt.plot(losses,color='blue', marker='s', label='训练')
# # plt.show()
#
# plt.plot(accs,color='blue', marker='s', label='测试')
# plt.show()

class mnist_square():
    def __init__(self,train_db,lr=1e-2):
        self.lr = lr
        self.accs, self.losses = [], []
        self.train_db = train_db
        self.w1, self.b1 = tf.Variable(tf.random.normal([784, 256], stddev=0.1)), tf.Variable(tf.zeros([256]))
        self.w2, self.b2 = tf.Variable(tf.random.normal([256, 128], stddev=0.1)), tf.Variable(tf.zeros([128]))
        self.w3, self.b3 = tf.Variable(tf.random.normal([128, 10], stddev=0.1)), tf.Variable(tf.zeros([10]))

    def __forward(self,x):
        h1 = x @ self.w1 + self.b1
        h1 = tf.nn.relu(h1)
        h2 = h1 @ self.w2 + self.b2
        h2 = tf.nn.relu(h2)
        out = h2 @ self.w3 + self.b3
        return out

    def train(self):
        for _ in tqdm(range(20)):
            for step, (x, y) in tqdm(enumerate(train_db)):
                with tf.GradientTape() as tape:
                    out = self.__forward(x)
                    loss = tf.reduce_mean(tf.square(y - out))
                grads = tape.gradient(loss, [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3])
                for w, g in zip([self.w1, self.b1, self.w2, self.b2, self.w3, self.b3], grads):
                    w.assign_sub(self.lr * g)
                if (step + 1) % 80 == 0:
                    self.losses.append(loss.numpy())
                    pred = tf.argmax(out, axis=1)
                    label = tf.argmax(y, axis=1)
                    correct = tf.reduce_mean(tf.cast(tf.equal(pred,label),dtype=tf.float32))
                    self.accs.append(correct.numpy())


(x, y), (x_val, y_val) = datasets.mnist.load_data()
train_db = tf.data.Dataset.from_tensor_slices((x,y)).shuffle(10000).batch(128).map(process)
model = mnist_square(train_db)
model.train()

