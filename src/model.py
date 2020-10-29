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

    def _conv_layers(self, chm_input, rgb_input, hsi_input, las_input):
        # Image network
        image_net = layers.Concatenate(axis=3)([hsi_input, chm_input, rgb_input])
        image_net = layers.Conv2D(7, 5, activation="relu", name="img_1")(image_net)
        image_net = layers.Conv2D(8, 5, activation="relu", name="img_2")(image_net)
        image_net = layers.Conv2D(8, 5, activation="relu", name="img_3")(image_net)
        image_net = layers.Conv2D(8, 5, activation="relu", name="img_4")(image_net)
        image_net = layers.Conv2D(16, 5, activation="relu", name="img_5")(image_net)
        image_net = layers.Conv2D(16, 5, activation="relu", name="img_6")(image_net)
        image_net = layers.Conv2D(32, 5, activation="relu", name="img_7")(image_net)
        image_net = layers.Conv2D(32, 5, activation="relu", name="img_8")(image_net)

        # Las 3D network
        las_net = layers.Conv3D(2, 3, activation="relu", name="las_1")(las_input)
        las_net = layers.Conv3D(2, 3, strides=(1, 1, 2), activation="relu", name="las_2")(las_net)
        las_net = layers.Conv3D(8, 3, activation="relu", name="las_3")(las_net)
        las_net = layers.Conv3D(8, 3, strides=(1, 1, 2), activation="relu", name="las_4")(las_net)
        las_net = layers.Conv3D(16, 3, activation="relu", name="las_5")(las_net)
        las_net = layers.Conv3D(16, 3, strides=(1, 1, 2), activation="relu", name="las_6")(las_net)
        las_net = layers.Conv3D(32, 3, activation="relu", name="las_7")(las_net)
        las_net = layers.Conv3D(32, (3, 3, 4), activation="relu", name="las_8")(las_net)
        las_net = layers.Reshape((24, 24, 32), input_shape=(24, 24, 1, 32))(las_net) 

        return (image_net, las_net)

    def _get_window(self, features, row, col):
        y = col * 5
        x = row * 5

    def build_rpn(self):
        chm_input = Input(shape=(200, 200, 1), name="chm")
        rgb_input = Input(shape=(200, 200, 3), name="rgb")
        hsi_input = Input(shape=(200, 200, 3), name="hsi")
        las_input = Input(shape=(40, 40, 70, 1), name="las")

        image_net, las_net = self._conv_layers(chm_input, rgb_input, hsi_input, las_input)

        las_up = layers.UpSampling2D(size=7)(las_net)
        features = layers.Concatenate(axis=3)([las_up, image_net])
        las_net = layers.Flatten()(las_net)
        image_net = layers.Flatten()(image_net)
        #image_net = layers.Cropping2D(((x, y), (x, y)))(image_net)

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
