import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import Model, Input
from tensorflow import keras

class LeafNet():
    def __init__(self):
        chm_input = Input(shape=(20, 20, 1), name="chm")
        rgb_input = Input(shape=(200, 200, 3), name="rgb")
        hsi_input = Input(shape=(20, 20, 369), name="hsi")
        las_input = Input(shape=(40, 40, 70, 1), name="las")

        # RGB downsample network
        rgb_down = layers.Conv2D(32, 5, activation="relu")(rgb_input)
        rgb_down = layers.Conv2D(32, 5, activation="relu")(rgb_down)
        rgb_down = layers.MaxPool2D(2)(rgb_down)
        rgb_down = layers.Conv2D(32, 5, activation="relu")(rgb_down)
        rgb_down = layers.Conv2D(32, 5, activation="relu")(rgb_down)
        rgb_down = layers.MaxPool2D(2)(rgb_down)
        rgb_down = layers.Conv2D(32, 5, activation="relu")(rgb_down)
        rgb_down = layers.MaxPool2D(2)(rgb_down)

        # HSI upsample network
        hsi_up = layers.Conv2D(256, 2, activation="relu", padding="same")(hsi_input)
        hsi_up = layers.UpSampling2D(3)(hsi_up)
        hsi_up = layers.Conv2D(128, 4, activation="relu")(hsi_up)
        hsi_up = layers.Conv2D(64, 4, activation="relu")(hsi_up)
        hsi_up = layers.UpSampling2D(2)(hsi_up)
        hsi_up = layers.Conv2D(32, 5, activation="relu")(hsi_up)
        hsi_up = layers.Conv2D(16, 5, activation="relu")(hsi_up)
        hsi_up = layers.UpSampling2D(2)(hsi_up)

        # CHM upsample network
        chm_up = layers.Conv2D(1, 2, activation="relu", padding="same")(chm_input)
        chm_up = layers.UpSampling2D(3)(chm_up)
        chm_up = layers.Conv2D(1, 4, activation="relu")(chm_up)
        chm_up = layers.Conv2D(1, 4, activation="relu")(chm_up)
        chm_up = layers.UpSampling2D(2)(chm_up)
        chm_up = layers.Conv2D(1, 5, activation="relu")(chm_up)
        chm_up = layers.Conv2D(1, 5, activation="relu")(chm_up)
        chm_up = layers.UpSampling2D(2)(chm_up)

        # High-res network
        high_res = tf.concat([rgb_input, hsi_up, chm_up], 3)
        high_res = layers.Conv2D(20, 5, activation="relu")(high_res)
        high_res = layers.Conv2D(10, 5, activation="relu")(high_res)
        high_res = layers.Conv2D(5, 5, activation="relu")(high_res)
        high_res = layers.Flatten()(high_res)

        # Low-res network
        low_res = layers.Conv2D(256, 2, activation="relu", padding="same")(hsi_input)
        low_res = layers.Conv2D(128, 2, activation="relu", padding="same")(low_res)
        low_res = layers.Conv2D(64, 2, activation="relu", padding="same")(low_res)
        low_res = layers.Conv2D(32, 2, activation="relu", padding="same")(low_res)
        low_res = tf.concat([low_res, chm_input, rgb_down], 3)
        low_res = layers.Conv2D(64, 2, activation="relu", padding="same")(low_res)
        low_res = layers.Conv2D(128, 2, activation="relu", padding="same")(low_res)
        low_res = layers.Conv2D(256, 2, activation="relu", padding="same")(low_res)
        low_res = layers.Flatten()(low_res)

        # Las 3D network
        las_net = layers.Conv3D(2, 16, activation="relu", padding="same")(las_input)
        las_net = layers.Conv3D(2, 32, activation="relu", padding="same")(las_net)
        las_net = layers.Conv3D(2, 64, activation="relu", padding="same")(las_net)
        las_net = layers.Conv3D(2, 128, activation="relu", padding="same")(las_net)
        las_net = layers.Flatten()(las_net)

        # Combine networks with fully connected layers
        fully_con = layers.concatenate([high_res, low_res, las_net])
        fully_con = layers.Dense(1024)(fully_con)
        output_bounding = layers.Dense(120)(fully_con)
        output_bounding = layers.Reshape((30, 4), name="bounds")(output_bounding)
        output_class = layers.Dense(30, name="labels")(fully_con)

        self.model = Model(
            inputs=[rgb_input, chm_input, hsi_input, las_input],
            outputs=[output_bounding, output_class],
        )

    def plot(self):
        keras.utils.plot_model(self.model, "leafnet.png", show_shapes=True)

    def compile(self):
        self.model.compile(
            loss=[keras.losses.MeanSquaredError(), keras.losses.BinaryCrossentropy()],
            optimizer=keras.optimizers.RMSprop(),
            metrics=["mse", "binary_accuracy"],
        )

    def fit(self, data_sequence):
        self.model.fit(
            data_sequence,
            epochs=100,
            batch_size=1,
        )
