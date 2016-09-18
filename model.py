import tensorflow as tf
import numpy as np
import depth_net_utils as utils
import matplotlib.pyplot as plt


class Model:
    def __init__(self, params_path, img_dem_row, img_dem_col):
        self.img_dem_row = img_dem_row
        self.img_dem_col = img_dem_col
        self.current_stage = ''
        self.weights = utils.get_weights(params_path + "/weights")
        self.biases = utils.get_weights(params_path + "/biases")
        self.output = 0
        self.session = None
        self.model_output = 0
        self.vars_initialized = False
        self.model_input = tf.placeholder(tf.float32, shape=(self.img_dem_row, self.img_dem_col), name="x")
        self.__build_model()

    def __get_weights(self, name):
        current_stage_weights = self.weights[self.current_stage]
        return tf.constant(current_stage_weights[name], name='weights')

    def __get_biases(self, name):
        current_stage_biases = self.biases[self.current_stage]
        return tf.constant(current_stage_biases[name], name='biases')

    def __conv_layer(self, _x, name):
        with tf.variable_scope(self.current_stage):
            kernel = self.__get_weights(name)
            bias = self.__get_biases(name)

            conv = tf.nn.conv2d(_x, kernel, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, bias)

            return tf.nn.relu(bias)

    def __conv_block(self, block_input, layer_names):
        self.output = block_input
        for i in layer_names:
            self.output = self.__conv_layer(self.output, i)
        return self.output

    def __max_pool(self, bottom, name):
        with tf.variable_scope(self.current_stage):
            return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def __fc_layer(self, _x, name):
        with tf.variable_scope(self.current_stage):
            shape = _x.get_shape().as_list()
            weights = self.__get_weights(name)
            biases = self.__get_biases(name)

            die = 1
            for dem_d in shape[1:]:
                die *= dem_d
            _x = tf.reshape(_x, [-1, die])

        return tf.nn.bias_add(tf.matmul(_x, weights), biases)

    def __build_vgg_model(self):
        """
        Model Architecture = conv_block1 -> max_pool -> conv_block2 -> max_pool -> conv_block3 -> max_pool ->
        conv_block4 -> max_pool -> conv_block5 -> FC1 -> FC2
        """
        self.current_stage = "vgg"
        vgg_input = tf.image.resize_bilinear(self.model_input, size=(112, 150))

        conv1 = self.__conv_block(vgg_input, ["conv1_1, conv1_2"])
        pool1 = self.__max_pool(conv1, 'pool1')

        conv2 = self.__conv_block(pool1, ['conv2_1', 'conv2_2'])
        pool2 = self.__max_pool(conv2, 'pool2')

        conv3 = self.__conv_block(pool2, ['conv3_1', 'conv3_2'])
        pool3 = self.__max_pool(conv3, 'pool3')

        conv4 = self.__conv_block(pool3, ['conv4_1', 'conv4_2', 'conv4_3'])
        pool4 = self.__max_pool(conv4, 'pool4')

        conv5 = self.__conv_block(pool4, ['conv5_1', 'conv5_2', 'conv5_3'])
        pool5 = self.__max_pool(conv5, 'pool5')

        fc1 = self.__fc_layer(pool5, 'fc1')
        assert fc1.get_shape().as_list()[1:] == [4096]
        relu_fc1 = tf.nn.relu(fc1)

        fc2 = self.__fc_layer(relu_fc1, 'fc2')
        relu_fc2 = tf.nn.relu(fc2)

        self.output = tf.image.resize_bilinear(relu_fc2, size=(55, 74))

    def __build_scale2_model(self):
        vgg_output = self.output
        scale2_input = tf.image.resize_bilinear(self.model_input, size=(55, 74))
        self.current_stage = "scale2"

        conv1 = self.__conv_layer(tf.add(scale2_input, vgg_output), 'conv1')

        conv2 = self.__conv_layer(conv1, 'conv2')

        conv3 = self.__conv_layer(conv2, 'conv3')

        conv4 = self.__conv_layer(conv3, 'conv4')

        conv5 = self.__conv_layer(conv4, 'conv5')

        self.output = tf.image.resize_bilinear(conv5, size=(109, 147))

    def __build_scale3_model(self):
        scale2_output = self.output
        scale3_input = tf.image.resize_bilinear(self.model_input, size=(109, 147))
        self.current_stage = "scale3"

        conv1 = self.__conv_layer(tf.add(scale3_input, scale2_output), 'conv1')

        conv2 = self.__conv_layer(conv1, 'conv2')

        conv3 = self.__conv_layer(conv2, 'conv3')

        self.model_output = self.__conv_layer(conv3, 'conv4')

    def __build_model(self):
        self.__build_vgg_model()
        self.__build_scale2_model()
        self.__build_scale3_model()

    def __init_tensorflow_session(self):
        if self.session is None:
            self.session = tf.Session()

    def __initialize_variables(self):
        if not self.vars_initialized:
            self.session.run(tf.initialize_all_variables())

    def save_model(self, name):
        self.__init_tensorflow_session()
        saver = tf.train.Saver()
        saver.save(self.session, name)

    def predict_depth(self, img):
        self.__init_tensorflow_session()
        self.__initialize_variables()
        self.session.run(self.model_output, feed_dict={self.model_input: img})
