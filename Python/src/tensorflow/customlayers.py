
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import Model, layers
from tensorflow.keras.datasets import mnist

# To Avoid GPU errors
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

class OurOwnCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get("accuracy") > 0.99:
            print("Accuracy over 99%, quitting training")
            self.model.stop_training = True

class Dense(layers.Layer):
    def __init__(self, units, input_dim):
        super(Dense, self).__init__()
        self.w = self.add_weight(
            name="w",
            shape=(input_dim, units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            name="b", shape=(units,), initializer="zeros", trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


class Dense(layers.Layer):
    def __init__(self, units):
        super(Dense, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            name="w",
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            name="b", shape=(self.units,), initializer="random_normal", trainable=True,
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


class MyReLU(layers.Layer):
    def __init__(self):
        super(MyReLU, self).__init__()

    def call(self, x):
        return tf.math.maximum(x, 0)


class MyModel(Model):
    def __init__(self, num_classes=10):
        super(MyModel, self).__init__()
        self.dense1 = Dense(64)
        self.dense2 = Dense(num_classes)
        self.relu = MyReLU()

    def call(self, x):
        x = self.relu(self.dense1(x))
        return self.dense2(x)


class CustomLayers:
    def __init__(self):
        pass

    def run(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0
        x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255.0

        model = MyModel()
        model.load_weights('checkpoint/custom_layers/')
        model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=keras.optimizers.Adam(),
            metrics=["accuracy"],
        )

        save_callback = keras.callbacks.ModelCheckpoint(
            "checkpoint/custom_layers/", save_weights_only=True, monitor="train_acc", save_best_only=False,
        )

        lr_scheduler = keras.callbacks.ReduceLROnPlateau(
            monitor="loss", factor=0.1, patience=3, mode="max", verbose=1
        )

        model.fit(x_train, y_train, batch_size=32, epochs=2, callbacks=[save_callback, lr_scheduler, OurOwnCallback()], verbose=2)
        model.evaluate(x_test, y_test, batch_size=32, verbose=2)
        model.save("saved_model/custom_layers/")