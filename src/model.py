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
        chm_input = Input(shape=(40, 40, 1), name="chm")
        rgb_input = Input(shape=(40, 40, 3), name="rgb")
        hsi_input = Input(shape=(40, 40, 3), name="hsi")
        las_input = Input(shape=(8, 8, 70, 1), name="las")

        # Image network
        image_net = layers.Concatenate(axis=3)([hsi_input, chm_input, rgb_input])
        image_net = layers.Conv2D(4, 4, activation="relu", padding="same")(image_net)
        image_net = layers.Conv2D(8, 4, activation="relu", padding="same")(image_net)
        image_net = layers.Conv2D(8, 4, activation="relu", padding="same")(image_net)
        image_net = layers.MaxPool2D(2)(image_net)
        image_net = layers.Conv2D(16, 4, activation="relu", padding="same")(image_net)
        image_net = layers.Conv2D(16, 4, activation="relu", padding="same")(image_net)
        image_net = layers.MaxPool2D(2)(image_net)
        image_net = layers.Conv2D(32, 4, activation="relu", padding="same")(image_net)
        image_net = layers.Conv2D(32, 4, activation="relu", padding="same")(image_net)
        image_net = layers.MaxPool2D(2)(image_net)
        image_net = layers.Conv2D(64, 4, activation="relu", padding="same")(image_net)
        image_net = layers.Conv2D(64, 4, activation="relu", padding="same")(image_net)
        image_net = layers.Flatten()(image_net)

        # Las 3D network
        las_net = layers.Conv3D(2, 2, activation="relu", padding="same")(las_input)
        las_net = layers.Conv3D(2, 2, activation="relu", padding="same")(las_net)
        las_net = layers.Conv3D(8, 2, activation="relu", padding="same")(las_net)
        las_net = layers.Conv3D(8, 2, activation="relu", padding="same")(las_net)
        las_net = layers.MaxPool3D(2)(las_net)
        las_net = layers.Conv3D(16, 2, activation="relu", padding="same")(las_net)
        las_net = layers.Conv3D(16, 4, activation="relu", padding="same")(las_net)
        las_net = layers.Conv3D(32, 2, activation="relu", padding="same")(las_net)
        las_net = layers.Conv3D(32, 2, activation="relu", padding="same", name="las_net")(las_net)
        las_net = layers.Flatten()(las_net)

        # Combine networks with fully connected layers
        fully_con = layers.concatenate([image_net, las_net])
        fully_con = layers.Dropout(0.1)(fully_con)
        fully_con = layers.Dense(256)(fully_con)
        output_bounding = layers.Dense(36)(fully_con)
        output_bounding = layers.Reshape((9, 4), name="bounds")(output_bounding)
        output_class = layers.Dense(9, activation="sigmoid", name="labels")(fully_con)

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
