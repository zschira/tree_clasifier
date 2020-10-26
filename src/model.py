from keras import layers
from keras import Model, Input
import keras

class LeafNet():
    def __init__(self):
        self.model = None

    def load_weights(self, path):
        if self.model == None:
            self.build_model()
            self.compile()

        self.model.load_weights(path / "model.h5")

    def build_model(self):
        chm_input = Input(shape=(20, 20, 1), name="chm")
        rgb_input = Input(shape=(200, 200, 3), name="rgb")
        hsi_input = Input(shape=(20, 20, 3), name="hsi")
        las_input = Input(shape=(40, 40, 70, 1), name="las")

        # RGB downsample network
        rgb_down = layers.Conv2D(3, 5, activation="relu")(rgb_input)
        rgb_down = layers.Conv2D(3, 5, activation="relu")(rgb_down)
        rgb_down = layers.MaxPool2D(2)(rgb_down)
        rgb_down = layers.Conv2D(8, 5, activation="relu")(rgb_down)
        rgb_down = layers.Conv2D(8, 5, activation="relu")(rgb_down)
        rgb_down = layers.MaxPool2D(2)(rgb_down)
        rgb_down = layers.Conv2D(16, 4, activation="relu")(rgb_down)
        rgb_down = layers.Conv2D(16, 4, activation="relu")(rgb_down)
        rgb_down = layers.MaxPool2D(2, name="rgb_down")(rgb_down)
        rgb_down = layers.Conv2D(32, 4, activation="relu")(rgb_down)
        rgb_down = layers.Conv2D(32, 4, activation="relu")(rgb_down)
        rgb_down = layers.Flatten()(rgb_down)

        """
        # HSI upsample network
        hsi_up = layers.Conv2D(3, 2, activation="relu", padding="same")(hsi_input)
        hsi_up = layers.UpSampling2D(3)(hsi_up)
        #hsi_up = layers.Dropout(0.4)(hsi_up)
        hsi_up = layers.Conv2D(3, 4, activation="relu")(hsi_up)
        hsi_up = layers.Conv2D(3, 4, activation="relu")(hsi_up)
        hsi_up = layers.UpSampling2D(2)(hsi_up)
        #hsi_up = layers.Dropout(0.4)(hsi_up)
        hsi_up = layers.Conv2D(3, 5, activation="relu")(hsi_up)
        hsi_up = layers.Conv2D(3, 5, activation="relu")(hsi_up)
        hsi_up = layers.UpSampling2D(2, name="hsi_up")(hsi_up)
        #hsi_up = layers.Dropout(0.4)(hsi_up)

        # CHM upsample network
        chm_up = layers.Conv2D(1, 2, activation="relu", padding="same")(chm_input)
        chm_up = layers.UpSampling2D(3)(chm_up)
        #chm_up = layers.Dropout(0.4)(chm_up)
        chm_up = layers.Conv2D(1, 4, activation="relu")(chm_up)
        chm_up = layers.Conv2D(1, 4, activation="relu")(chm_up)
        chm_up = layers.UpSampling2D(2)(chm_up)
        #chm_up = layers.Dropout(0.4)(chm_up)
        chm_up = layers.Conv2D(1, 5, activation="relu")(chm_up)
        chm_up = layers.Conv2D(1, 5, activation="relu")(chm_up)
        chm_up = layers.UpSampling2D(2, name="chm_up")(chm_up)
        #chm_up = layers.Dropout(0.4)(chm_up)

        # High-res network
        high_res = layers.Concatenate(axis=3)([rgb_input, hsi_up, chm_up])
        high_res = layers.Conv2D(10, 5, activation="relu")(high_res)
        high_res = layers.Conv2D(10, 5, activation="relu")(high_res)
        high_res = layers.Conv2D(5, 5, activation="relu", name="high_res")(high_res)
        high_res = layers.Flatten()(high_res)

        """
        # Low-res network
        low_res = layers.Concatenate(axis=3)([hsi_input, chm_input])
        low_res = layers.Conv2D(4, 2, activation="relu", padding="same")(low_res)
        low_res = layers.Conv2D(8, 2, activation="relu", padding="same")(low_res)
        low_res = layers.Conv2D(8, 2, activation="relu", padding="same")(low_res)
        low_res = layers.MaxPool2D(2)(low_res)
        low_res = layers.Conv2D(16, 2, activation="relu", padding="same")(low_res)
        low_res = layers.Conv2D(16, 2, activation="relu", padding="same")(low_res)
        low_res = layers.MaxPool2D(2)(low_res)
        low_res = layers.Conv2D(32, 2, activation="relu", padding="same")(low_res)
        low_res = layers.Conv2D(32, 2, activation="relu", padding="same")(low_res)
        low_res = layers.MaxPool2D(2)(low_res)
        low_res = layers.Conv2D(64, 2, activation="relu", padding="same")(low_res)
        low_res = layers.Conv2D(64, 2, activation="relu", padding="same")(low_res)
        low_res = layers.Flatten()(low_res)

        # Las 3D network
        las_net = layers.Conv3D(2, 4, activation="relu", padding="same")(las_input)
        las_net = layers.Conv3D(2, 4, activation="relu", padding="same")(las_net)
        las_net = layers.MaxPool3D(2)(las_net)
        las_net = layers.Conv3D(8, 4, activation="relu", padding="same")(las_net)
        las_net = layers.Conv3D(8, 4, activation="relu", padding="same")(las_net)
        las_net = layers.MaxPool3D(2)(las_net)
        las_net = layers.Conv3D(16, 4, activation="relu", padding="same")(las_net)
        las_net = layers.Conv3D(16, 4, activation="relu", padding="same")(las_net)
        las_net = layers.MaxPool3D(2)(las_net)
        las_net = layers.Conv3D(32, 4, activation="relu", padding="same")(las_net)
        las_net = layers.Conv3D(32, 4, activation="relu", padding="same", name="las_net")(las_net)
        las_net = layers.Flatten()(las_net)

        # Combine networks with fully connected layers
        fully_con = layers.concatenate([low_res, las_net, rgb_down])
        fully_con = layers.Dropout(0.1)(fully_con)
        fully_con = layers.Dense(256)(fully_con)
        fully_con = layers.Dropout(0.4)(fully_con)
        fully_con = layers.Dense(256)(fully_con)
        fully_con = layers.Dropout(0.4)(fully_con)
        fully_con = layers.Dense(256)(fully_con)
        fully_con = layers.Dropout(0.4)(fully_con)
        fully_con = layers.Dense(256)(fully_con)
        fully_con = layers.Dropout(0.0)(fully_con)
        output_bounding = layers.Dense(120, kernel_regularizer=keras.regularizers.l2(0.0001))(fully_con)
        output_bounding = layers.Reshape((30, 4), name="bounds")(output_bounding)
        output_class = layers.Dense(30, activation="sigmoid", kernel_regularizer=keras.regularizers.l2(0.0001), name="labels")(fully_con)

        self.model = Model(
            inputs=[rgb_input, chm_input, hsi_input, las_input],
            outputs=[output_bounding, output_class],
        )

        self.model.summary()

    def plot(self):
        keras.utils.plot_model(self.model, "leafnet.png", show_shapes=True)

    def compile(self):
        if self.model == None:
            self.build_model()

        self.model.compile(
            loss={"bounds": "mae", "labels": "binary_crossentropy"},
            optimizer=keras.optimizers.RMSprop(),
            metrics={"bounds": "mse", "labels": "binary_accuracy"},
        )

    def fit(self, data_sequence, path):
        self.model.fit_generator(
            data_sequence,
            epochs=500,
        )

        self.model.save_weights(path / "model.h5")

    def predict(self, data_sequence):
        return self.model.predict(data_sequence)
