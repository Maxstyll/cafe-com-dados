import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.datasets import cifar10

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

class OurOwnCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get("accuracy") > 0.99:
            print("Accuracy over 99%, quitting training")
            self.model.stop_training = True

class Regularization:
    def __init__(self):
        pass

    def my_model(self):
        inputs = keras.Input(shape=(32, 32, 3))
        x = layers.Conv2D(32, 3, padding="same", kernel_regularizer=regularizers.l2(0.01),)(
            inputs
        )
        x = layers.BatchNormalization()(x)
        x = keras.activations.relu(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(64, 3, padding="same", kernel_regularizer=regularizers.l2(0.01),)(
            x
        )
        x = layers.BatchNormalization()(x)
        x = keras.activations.relu(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(
            128, 3, padding="same", kernel_regularizer=regularizers.l2(0.01),
        )(x)
        x = layers.BatchNormalization()(x)
        x = keras.activations.relu(x)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.01),)(
            x
        )
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(10)(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model

    def run(self):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0

        model = self.my_model()
        model.load_weights('checkpoint/regularization/')
        
        model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=keras.optimizers.Adam(lr=3e-4),
            metrics=["accuracy"],
        )


        save_callback = keras.callbacks.ModelCheckpoint(
            "checkpoint/regularization/", save_weights_only=True, monitor="train_acc", save_best_only=False,
        )

        lr_scheduler = keras.callbacks.ReduceLROnPlateau(
            monitor="loss", factor=0.1, patience=3, mode="max", verbose=1
        )

        model.fit(x_train, y_train, batch_size=64, epochs=150, callbacks=[save_callback, lr_scheduler, OurOwnCallback()], verbose=2)
        model.evaluate(x_test, y_test, batch_size=64, verbose=2)
        model.save("saved_model/regularization/")
