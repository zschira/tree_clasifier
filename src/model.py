from tensorflow.keras import layers
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Layer
import tensorflow as tf

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
        
        roi_input = Input(shape=(625 * 9, 4), name="roi_innie")

        image_net, las_net = self._conv_layers(chm_input, rgb_input, hsi_input, las_input)
        las_up = layers.UpSampling2D(size=7)(las_net)
        features = layers.Concatenate(axis=3)([las_up, image_net])
        roi_pool = ROIPoolingLayer(5, 5)([features, roi_input])
        roi_pool = layers.Flatten()(roi_pool)

        fc = layers.Dense(30)(roi_pool)

        """
        labels = layers.Dense(30)(fc, activation="sigmoid")
        bounds = layers.Dense(120)(fc)
        bounds = layers.Reshape((30, 4))(bounds)
        """

        self.detector = Model(
            inputs=[rgb_input, chm_input, hsi_input, las_input],
            outputs=[roi_pool],
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


class ROIPoolingLayer(Layer):
    """ Implements Region Of Interest Max Pooling 
        for channel-first images and relative bounding box coordinates
        
        # Constructor parameters
            pooled_height, pooled_width (int) -- 
              specify height and width of layer outputs
        
        Shape of inputs
            [(batch_size, pooled_height, pooled_width, n_channels),
             (batch_size, num_rois, 4)]
           
        Shape of output
            (batch_size, num_rois, pooled_height, pooled_width, n_channels)
    
    """
    def __init__(self, pooled_height, pooled_width, **kwargs):
        self.pooled_height = pooled_height
        self.pooled_width = pooled_width
        
        super(ROIPoolingLayer, self).__init__(**kwargs)
        
    def compute_output_shape(self, input_shape):
        """ Returns the shape of the ROI Layer output
        """
        feature_map_shape, rois_shape = input_shape
        assert feature_map_shape[0] == rois_shape[0]
        batch_size = feature_map_shape[0]
        n_rois = rois_shape[1]
        n_channels = feature_map_shape[3]
        return (batch_size, n_rois, self.pooled_height, 
                self.pooled_width, n_channels)

    def call(self, x):
        """ Maps the input tensor of the ROI layer to its output
        
            # Parameters
                x[0] -- Convolutional feature map tensor,
                        shape (batch_size, pooled_height, pooled_width, n_channels)
                x[1] -- Tensor of region of interests from candidate bounding boxes,
                        shape (batch_size, num_rois, 4)
                        Each region of interest is defined by four relative 
                        coordinates (x_min, y_min, x_max, y_max) between 0 and 1
            # Output
                pooled_areas -- Tensor with the pooled region of interest, shape
                    (batch_size, num_rois, pooled_height, pooled_width, n_channels)
        """
        def curried_pool_rois(x): 
          return ROIPoolingLayer._pool_rois(x[0], x[1], 
                                            self.pooled_height, 
                                            self.pooled_width)
        
        pooled_areas = tf.map_fn(curried_pool_rois, x, dtype=tf.float32)

        return pooled_areas
    
    @staticmethod
    def _pool_rois(feature_map, rois, pooled_height, pooled_width):
        """ Applies ROI pooling for a single image and varios ROIs
        """
        def curried_pool_roi(roi): 
          return ROIPoolingLayer._pool_roi(feature_map, roi, 
                                           pooled_height, pooled_width)
        
        pooled_areas = tf.map_fn(curried_pool_roi, rois, dtype=tf.float32)
        return pooled_areas
    
    @staticmethod
    def _pool_roi(feature_map, roi, pooled_height, pooled_width):
        """ Applies ROI pooling to a single image and a single region of interest
        """

        # Compute the region of interest        
        feature_map_height = int(feature_map.shape[0])
        feature_map_width  = int(feature_map.shape[1])
        
        h_start = tf.cast(feature_map_height * roi[0], 'int32')
        w_start = tf.cast(feature_map_width  * roi[1], 'int32')
        h_end   = tf.cast(feature_map_height * roi[2], 'int32')
        w_end   = tf.cast(feature_map_width  * roi[3], 'int32')
        
        region = feature_map[h_start:h_end, w_start:w_end, :]
        
        # Divide the region into non overlapping areas
        region_height = h_end - h_start
        region_width  = w_end - w_start
        h_step = tf.cast( region_height / pooled_height, 'int32')
        w_step = tf.cast( region_width  / pooled_width , 'int32')
        
        areas = [[(
                    i*h_step, 
                    j*w_step, 
                    (i+1)*h_step if i+1 < pooled_height else region_height, 
                    (j+1)*w_step if j+1 < pooled_width else region_width
                   ) 
                   for j in range(pooled_width)] 
                  for i in range(pooled_height)]
        
        # take the maximum of each area and stack the result
        def pool_area(x): 
          return tf.math.reduce_max(region[x[0]:x[2], x[1]:x[3], :], axis=[0,1])
        
        pooled_features = tf.stack([[pool_area(x) for x in row] for row in areas])
        return pooled_features
