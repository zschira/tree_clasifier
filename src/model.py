from keras import layers
from keras import Model, Input
import keras
import keras.backend as K

class LeafNet():
    def __init__(self):
        self.rpn = None

    def load_rpn_weights(self, path):
        if self.rpn == None:
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
        image_net = layers.Conv2D(16, 5, activation="relu", name="img_7")(image_net)
        image_net = layers.Conv2D(16, 5, activation="relu", name="img_8")(image_net)

        # Las 3D network
        las_net = layers.Conv3D(2, 3, activation="relu", name="las_1")(las_input)
        las_net = layers.Conv3D(2, 3, strides=(1, 1, 2), activation="relu", name="las_2")(las_net)
        las_net = layers.Conv3D(4, 3, activation="relu", name="las_3")(las_net)
        las_net = layers.Conv3D(4, 3, strides=(1, 1, 2), activation="relu", name="las_4")(las_net)
        las_net = layers.Conv3D(8, 3, activation="relu", name="las_5")(las_net)
        las_net = layers.Conv3D(8, 3, strides=(1, 1, 2), activation="relu", name="las_6")(las_net)
        las_net = layers.Conv3D(16, 3, activation="relu", name="las_7")(las_net)
        las_net = layers.Conv3D(16, (3, 3, 4), activation="relu", name="las_8")(las_net)
        las_net = layers.Reshape((24, 24, 16), input_shape=(24, 24, 1, 16))(las_net) 

        return (image_net, las_net)

    def _get_window(self, features, row, col):
        left = col * 6
        right = 168 - ((col * 6) + 24)
        top = row * 6
        bot = 168 - ((row * 6) + 24)
        feature_map = layers.Cropping2D(((top, bot), (left, right)))(features)

        return feature_map

    def build_rpn(self):
        chm_input = Input(shape=(200, 200, 1), name="chm")
        rgb_input = Input(shape=(200, 200, 3), name="rgb")
        hsi_input = Input(shape=(200, 200, 3), name="hsi")
        las_input = Input(shape=(40, 40, 70, 1), name="las")

        image_net, las_net = self._conv_layers(chm_input, rgb_input, hsi_input, las_input)

        las_up = layers.UpSampling2D(size=7)(las_net)
        features = layers.Concatenate(axis=3)([las_up, image_net])

        dense = layers.Dense(256)
        bound_layer = layers.Dense(36)
        reshape_bounds = layers.Reshape((9, 4))
        class_layer = layers.Dense(9, activation="sigmoid")
        conv1 = layers.Conv2D(4, 1)

        bounds_list = []
        labels_list = []
        for row in range(25):
            for col in range(25):
                window = self._get_window(features, row, col)
                window = layers.Flatten()(conv1(window))
                feature_dense = dense(window)
                bounds = bound_layer(feature_dense)
                bounds = reshape_bounds(bounds)
                labels = class_layer(feature_dense)
                bounds_list.append(bounds)
                labels_list.append(labels)

        output_bounding = layers.Concatenate(axis=1, name="bounds")(bounds_list)
        output_class = layers.Concatenate(axis=1, name="labels")(labels_list)

        self.rpn = Model(
            inputs=[rgb_input, chm_input, hsi_input, las_input],
            outputs=[output_bounding, output_class],
        )

        self.rpn.summary()

    def build_detector(self):
        chm_input = Input(shape=(200, 200, 1), name="chm")
        rgb_input = Input(shape=(200, 200, 3), name="rgb")
        hsi_input = Input(shape=(200, 200, 3), name="hsi")
        las_input = Input(shape=(40, 40, 70, 1), name="las")
        
        roi_input = Input(shape=(625 * 9, 4), name="roi")

        image_net, las_net = self._conv_layers(chm_input, rgb_input, hsi_input, las_input)
        las_up = layers.UpSampling2D(size=7)(las_net)
        features = layers.Concatenate(axis=3)([las_up, image_net])
        roi_pool = RoiPoolingConv(5, 625 * 9)([features, roi_input])
        roi_pool = layers.Flatten()(roi_pool)

        fc = layers.Dense(256)(roi_pool)

        labels = layers.Dense(30)(fc, activation="sigmoid")
        bounds = layers.Dense(120)(fc)
        bounds = layers.Reshape((30, 4))(bounds)

        self.detector = Model(
            inputs=[rgb_input, chm_input, hsi_input, las_input],
            outputs=[bounds, labels],
        )

        self.detector.summary()

    def compile(self):
        if self.rpn == None:
            self.build_rpn()

        self.rpn.compile(
            loss={"bounds": "mae", "labels": "binary_crossentropy"},
            optimizer=keras.optimizers.RMSprop(),
            metrics={"bounds": "mse", "labels": "binary_accuracy"},
        )

    def fit(self, data_sequence, path):
        self.model.fit_generator(
            data_sequence,
            epochs=100,
        )

        self.model.save_weights(path / "model.h5")

    def predict(self, data_sequence):
        return self.model.predict(data_sequence)


class RoiPoolingConv(layers.Layer):
    '''ROI pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_size: int
            Size of pooling region to use. pool_size = 7 will result in a 7x7 region.
        num_rois: number of regions of interest to be used
    # Input shape
        list of two 4D tensors [X_img,X_roi] with shape:
        X_img:
        `(1, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(1, rows, cols, channels)` if dim_ordering='tf'.
        X_roi:
        `(1,num_rois,4)` list of rois, with ordering (x,y,w,h)
    # Output shape
        3D tensor with shape:
        `(1, num_rois, channels, pool_size, pool_size)`
    '''
    def __init__(self, pool_size, num_rois, **kwargs):

        self.dim_ordering = 'tf'
        assert self.dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'

        self.pool_size = pool_size
        self.num_rois = num_rois

        super(RoiPoolingConv, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.dim_ordering == 'th':
            self.nb_channels = input_shape[0][1]
        elif self.dim_ordering == 'tf':
            self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        if self.dim_ordering == 'th':
            return None, self.num_rois, self.nb_channels, self.pool_size, self.pool_size
        else:
            return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    def call(self, x, mask=None):

        assert(len(x) == 2)

        img = x[0]
        rois = x[1]

        input_shape = K.shape(img)

        outputs = []

        for roi_idx in range(self.num_rois):

            x = rois[0, roi_idx, 0]
            y = rois[0, roi_idx, 1]
            w = rois[0, roi_idx, 2]
            h = rois[0, roi_idx, 3]
            
            row_length = w / float(self.pool_size)
            col_length = h / float(self.pool_size)

            num_pool_regions = self.pool_size

            #NOTE: the RoiPooling implementation differs between theano and tensorflow due to the lack of a resize op
            # in theano. The theano implementation is much less efficient and leads to long compile times

            if self.dim_ordering == 'th':
                for jy in range(num_pool_regions):
                    for ix in range(num_pool_regions):
                        x1 = x + ix * row_length
                        x2 = x1 + row_length
                        y1 = y + jy * col_length
                        y2 = y1 + col_length

                        x1 = K.cast(x1, 'int32')
                        x2 = K.cast(x2, 'int32')
                        y1 = K.cast(y1, 'int32')
                        y2 = K.cast(y2, 'int32')

                        x2 = x1 + K.maximum(1,x2-x1)
                        y2 = y1 + K.maximum(1,y2-y1)
                        
                        new_shape = [input_shape[0], input_shape[1],
                                     y2 - y1, x2 - x1]

                        x_crop = img[:, :, y1:y2, x1:x2]
                        xm = K.reshape(x_crop, new_shape)
                        pooled_val = K.max(xm, axis=(2, 3))
                        outputs.append(pooled_val)

            elif self.dim_ordering == 'tf':
                x = K.cast(x, 'int32')
                y = K.cast(y, 'int32')
                w = K.cast(w, 'int32')
                h = K.cast(h, 'int32')

                rs = tf.image.resize(img[:, y:y+h, x:x+w, :], (self.pool_size, self.pool_size))
                outputs.append(rs)

        final_output = K.concatenate(outputs, axis=0)
        final_output = K.reshape(final_output, (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))

        if self.dim_ordering == 'th':
            final_output = K.permute_dimensions(final_output, (0, 1, 4, 2, 3))
        else:
            final_output = K.permute_dimensions(final_output, (0, 1, 2, 3, 4))

        return final_output
    
    
    def get_config(self):
        config = {'pool_size': self.pool_size,
                  'num_rois': self.num_rois}
        base_config = super(RoiPoolingConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

