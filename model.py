import tensorflow as tf
import tensorflow.contrib.slim as slim

variables_dict = {
    "hidden_Weights1": tf.Variable(tf.truncated_normal([11136, 2048], stddev=0.1), name="hidden_Weights1"),
    "hidden_Weights2": tf.Variable(tf.truncated_normal([2048, 128], stddev=0.1), name="hidden_Weights2"),
    "hidden_biases1": tf.Variable(tf.constant(0.1, shape=[2048]), name="hidden_biases1"),
    "hidden_biases2": tf.Variable(tf.constant(0.1, shape=[128]), name="hidden_biases2")
}


class SIAMESE(object):
    def siamesenet(self, input,is_training=None, reuse=False):
        with tf.name_scope("model"):
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
            with tf.variable_scope("conv4") as scope:
                net = slim.layers.conv2d(net, 256, [3, 3], activation_fn=tf.nn.relu, padding='SAME',
                                               weights_initializer=slim.xavier_initializer_conv2d(),
                                               scope=scope, reuse=reuse)

            with tf.variable_scope("conv5") as scope:
                net = slim.layers.conv2d(net, 256, [3, 3], activation_fn=tf.nn.relu, padding='SAME',
                                               weights_initializer=slim.xavier_initializer_conv2d(),
                                               scope=scope, reuse=reuse)
                # net = slim.layers.max_pool2d(net, [2, 2], padding='SAME')
                net = tf.nn.max_pool(net, [1, 2, 2, 1], strides=[1,2,2,1], padding='SAME', name="conv_4")
                output_0 = slim.layers.flatten(net)
            with tf.variable_scope("conv6") as scope:
                net = slim.layers.conv2d(net, 512, [3, 3], activation_fn=tf.nn.relu, padding='SAME',
                                               weights_initializer=slim.xavier_initializer_conv2d(),
                                               scope=scope, reuse=reuse)

            with tf.variable_scope("conv7") as scope:
                net = slim.layers.conv2d(net, 512, [3, 3], activation_fn=tf.nn.relu, padding='SAME',
                                               weights_initializer=slim.xavier_initializer_conv2d(),
                                               scope=scope, reuse=reuse)
                # net = slim.layers.max_pool2d(net, [2, 2], padding='SAME')
                net = tf.nn.max_pool(net, [1, 2, 2, 1], strides=[1,2,2,1], padding='SAME', name="conv_5")
                output_1 = slim.flatten(net)

            with tf.variable_scope("conv8") as scope:
                net = slim.layers.conv2d(net, 32, [3, 3], activation_fn=None, padding='SAME',
                                               weights_initializer=slim.xavier_initializer_conv2d(),
                                               scope=scope, reuse=reuse)
                # net = slim.layers.max_pool2d(net, [2, 2], padding='SAME')
                net = tf.nn.max_pool(net, [1, 2, 2, 1], strides=[1,2,2,1], padding='SAME', name="conv_6")
            output_2 = slim.layers.flatten(net)

            net = tf.concat([output_0, output_1, output_2], 1,name='concat')

            with tf.variable_scope("local1") as scope:
                net = tf.nn.relu(tf.matmul(net, variables_dict["hidden_Weights1"])+
                                 variables_dict["hidden_biases1"], name=scope.name)


            # add hidden layer1
            # hidden_Weights = tf.Variable(tf.truncated_normal([11136, 2048], stddev=0.1))
            # hidden_biases = tf.Variable(tf.constant(0.1, shape=[2048]))
            # # net = slim.dropout(net,is_training=True, keep_prob=1.0)
            # net = tf.nn.relu(tf.matmul(net, hidden_Weights) + hidden_biases,name="hidden_layer1")

            if is_training is not None:
                # 为了方便起见，这里训练和测试时候都取 1.0
                keep_prob = 1.0
            else :
                keep_prob = 1.0
            with tf.variable_scope("dropout") as scope:
                net = slim.layers.dropout(net, keep_prob=keep_prob, scope=scope)

            # add hidden layer2
            #hidden_Weights = tf.Variable(tf.truncated_normal([2048, 128], stddev=0.1))
            #hidden_biases = tf.Variable(tf.constant(0.1, shape=[128]))
            # net = slim.dropout(net, is_training=True, keep_prob=1.0)
            #net = tf.nn.relu(tf.matmul(net, hidden_Weights) + hidden_biases,name="final_out")
            with tf.variable_scope("local2") as scope:
                net = tf.nn.relu(tf.matmul(net, variables_dict["hidden_Weights2"])+
                                 variables_dict["hidden_biases2"], name=scope.name)
        return net

    def contrastive_loss(self, model1, model2, y):
        with tf.name_scope("output"):
            # 测试使用 #############################
            w_test = tf.Variable(tf.ones([128,128]), name='w_test')
            model1_test = tf.matmul(model1,w_test,name='model1_test')
            model2_test = tf.matmul(model2,w_test, name='model2_test')
            ######################################
            output_difference = tf.abs(model1 - model2)
            W = tf.Variable(tf.random_normal([128, 1], stddev=0.1), name='W')
            b = tf.Variable(tf.zeros([1, 1]) + 0.1, name='b')
            y_ = tf.nn.sigmoid(tf.matmul(output_difference, W) + b, name='distance')

        # CalculateMean loss
        with tf.name_scope("loss"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_)   # 交叉熵
            loss = tf.reduce_mean(losses)

        # Calculate Accuracy
        with tf.name_scope("accuracy"):
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(y_), y), tf.float32),name="accuracy")

        return model1, model2, y_, loss, accuracy

