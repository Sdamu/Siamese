import tensorflow as tf
from ResNet import *
import tensorflow.contrib.slim as slim

variables_dict = {
    "hidden_Weights": tf.Variable(tf.truncated_normal([4096, 128], stddev=0.1), name="hidden_Weights"),
    "hidden_biases": tf.Variable(tf.constant(0.1, shape=[128]), name="hidden_biases")
}


class SIAMESE(object):
    def siamesenet(self, input, reuse=False, is_training=False):
        with tf.name_scope("model"):
            # 使用原图像尺寸的大小
            with tf.variable_scope("conv1") as scope:
                net = slim.layers.conv2d(input, 32, [7, 7], activation_fn=tf.nn.relu, padding='SAME',
                                               weights_initializer=slim.xavier_initializer_conv2d(),
                                               scope=scope, reuse=reuse)

                # net = slim.layers.max_pool2d(net, [2, 2], padding='SAME')
                net = tf.nn.max_pool(net,[1, 2, 2, 1],strides=[1,2,2,1],padding='SAME',name="conv_1")
            with tf.variable_scope("conv2") as scope:
                net = slim.layers.conv2d(net, 64, [5, 5], activation_fn=tf.nn.relu, padding='SAME',
                                               weights_initializer=slim.xavier_initializer_conv2d(),
                                               scope=scope, reuse=reuse)
                # net = slim.layers.max_pool2d(net, [2, 2], padding='SAME')
                net = tf.nn.max_pool(net,[1, 2, 2, 1], strides=[1,2,2,1], padding='SAME', name="conv_2")
            with tf.variable_scope("conv3") as scope:
                net = slim.layers.conv2d(net, 128, [3, 3], activation_fn=tf.nn.relu, padding='SAME',
                                               weights_initializer=slim.xavier_initializer_conv2d(),
                                               scope=scope, reuse=reuse)
                # net = slim.layers.max_pool2d(net, [2, 2], padding='SAME')
                net = tf.nn.max_pool(net, [1, 2, 2, 1], strides=[1,2,2,1], padding='SAME', name="conv_3")
            # with tf.variable_scope("conv4") as scope:
            #     net = slim.layers.conv2d(net, 256, [3, 3], activation_fn=tf.nn.relu, padding='SAME',
            #                                    weights_initializer=slim.xavier_initializer_conv2d(),
            #                                    scope=scope, reuse=reuse)

            with tf.variable_scope("conv4") as scope:
                net = slim.layers.conv2d(net, 256, [3, 3], activation_fn=tf.nn.relu, padding='SAME',
                                               weights_initializer=slim.xavier_initializer_conv2d(),
                                               scope=scope, reuse=reuse)
                # net = slim.layers.max_pool2d(net, [2, 2], padding='SAME')
                net = tf.nn.max_pool(net, [1, 2, 2, 1], strides=[1,2,2,1], padding='SAME', name="conv_4")
                output_0 = slim.layers.flatten(net)
            # with tf.variable_scope("conv6") as scope:
            #     net = slim.layers.conv2d(net, 512, [3, 3], activation_fn=tf.nn.relu, padding='SAME',
            #                                    weights_initializer=slim.xavier_initializer_conv2d(),
            #                                    scope=scope, reuse=reuse)

            with tf.variable_scope("conv5") as scope:
                net = slim.layers.conv2d(net, 512, [3, 3], activation_fn=tf.nn.relu, padding='SAME',
                                               weights_initializer=slim.xavier_initializer_conv2d(),
                                               scope=scope, reuse=reuse)
                # net = slim.layers.max_pool2d(net, [2, 2], padding='SAME')
                net = tf.nn.max_pool(net, [1, 2, 2, 1], strides=[1,2,2,1], padding='SAME', name="conv_5")
                output_1 = slim.flatten(net)

            with tf.variable_scope("conv6") as scope:
                net = slim.layers.conv2d(net, 32, [3, 3], activation_fn=None, padding='SAME',
                                               weights_initializer=slim.xavier_initializer_conv2d(),
                                               scope=scope, reuse=reuse)
                # net = slim.layers.max_pool2d(net, [2, 2], padding='SAME')
                net = tf.nn.max_pool(net, [1, 2, 2, 1], strides=[1,2,2,1], padding='SAME', name="conv_6")
            output_2 = slim.layers.flatten(net)

            net = tf.concat([output_0, output_1, output_2], 1,name='concat')

            # add hidden layer1
            hidden_Weights = tf.Variable(tf.truncated_normal([11136, 2048], stddev=0.1))
            hidden_biases = tf.Variable(tf.constant(0.1, shape=[2048]))
            net = slim.dropout(net,is_training=True, keep_prob=0.5)
            net = tf.nn.relu(tf.matmul(net, hidden_Weights) + hidden_biases,name="hidden_layer1")

            with tf.variable_scope("dropout") as scope:
                net = slim.layers.dropout(net, keep_prob=0.7, scope=scope)

            # add hidden layer2
            hidden_Weights = tf.Variable(tf.truncated_normal([2048, 128], stddev=0.1))
            hidden_biases = tf.Variable(tf.constant(0.1, shape=[128]))
            net = slim.dropout(net, is_training=True, keep_prob=0.5)
            net = tf.nn.relu(tf.matmul(net, hidden_Weights) + hidden_biases,name="final_out")

            tf.nn.l2_normalize(net)

            output = tf.layers.dense(net,2048,activation=tf.nn.relu,
                                     kernel_initializer=slim.xavier_initializer_conv2d())
            output = tf.layers.dense(output, 128, activation=tf.nn.relu,
                                     kernel_initializer=slim.xavier_initializer_conv2d())
            # with tf.variable_scope("local") as scope:
            #     output = tf.nn.relu(tf.matmul(flatten, variables_dict["hidden_Weights"]) +
            #                         variables_dict["hidden_biases"], name=scope.name)

        return output

    def contrastive_loss(self, model1, model2, y):
        with tf.name_scope("output"):
            output_difference = tf.square(tf.abs(model1 - model2))
            # W = tf.Variable(tf.random_normal([128, 1], stddev=0.1), name='W')
            # b = tf.Variable(tf.zeros([1, 1]) + 0.1, name='b')
            # y_ = tf.sigmoid(tf.add(tf.matmul(output_difference, W), b, name='predict'))

        # CalculateMean loss
        with tf.name_scope("loss"):
            losses = y*0.5*tf.square(output_difference)+(1-y)*0.5*tf.maximum(tf.cast(0,tf.float32),1-output_difference)
            loss = tf.reduce_sum(losses)
            # losses = -(y * tf.log(y_ + MIN_NUMBER) + (1 - y) * tf.log(1 - y_ + MIN_NUMBER))  # 交叉熵
            # loss = tf.reduce_mean(losses)

        # Calculate Accuracy
        with tf.name_scope("accuracy"):
            y_pred = tf.sigmoid(y_)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(y_pred), y), tf.float32),name="accuracy")

        return model1, model2, y_, loss, accuracy

