import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

class OurOwnCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get("accuracy") > 0.99:
            print("Accuracy over 99%, quitting training")
            self.model.stop_training = True

class NeuralNetwork:
    def __init__(self):
        pass

    def run(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0
        x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255.0

        model = keras.Sequential()
        model.add(keras.Input(shape=(784)))
        model.add(layers.Dense(512, activation="relu"))
        model.add(layers.Dense(256, activation="relu", name="my_layer"))
        model.add(layers.Dense(10))

        model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            optimizer=keras.optimizers.Adam(lr=0.001),
            metrics=["accuracy"],
        )


        save_callback = keras.callbacks.ModelCheckpoint(
            "checkpoint/neural_network/", save_weights_only=True, monitor="train_acc", save_best_only=False,
        )

        lr_scheduler = keras.callbacks.ReduceLROnPlateau(
            monitor="loss", factor=0.1, patience=3, mode="max", verbose=1
        )

        model.fit(x_train, y_train, batch_size=32, epochs=5, callbacks=[save_callback, lr_scheduler, OurOwnCallback()], verbose=2)
        model.evaluate(x_test, y_test, batch_size=32, verbose=2)
        model.save("saved_model/neural_network/")