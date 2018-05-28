import cv2
from model import *
import tensorflow as tf
import numpy as np
from resize_image import resize_image

image_1 = cv2.imread('book_02.jpg',0)
image_1 = cv2.merge([image_1, image_1, image_1])
image_2 = cv2.imread('book_01.jpg',0)
image_2 = cv2.merge([image_2, image_2, image_2])


def process_image(image):
    image = resize_image(image)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    #
    array_image = np.asarray(np.asarray(image),dtype='float32')/255.
    array_image.resize((1, 72, 72, 3))
    return array_image

graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.Session(config=session_conf)

    # sess = tf.Session()
    with sess.as_default():
        saver = tf.train.import_meta_graph('checkpoint/conv_8_layers_model_10.ckpt.meta')
        saver.restore(sess, 'checkpoint/conv_8_layers_model_10.ckpt')


        is_training = graph.get_operation_by_name("is_training/is_training").outputs[0]
        # 分别取出两个输入层
        left = graph.get_operation_by_name("in/left").outputs[0]
        right = graph.get_operation_by_name("in/right").outputs[0]
        distance = graph.get_operation_by_name("output/distance").outputs[0]
        ww = graph.get_operation_by_name("output/W").outputs[0]
        _out = graph.get_operation_by_name("model/final_out").outputs[0]



        # 读入要寻找的图片
        left_image = process_image(image_1)
        # 循环读入检测到的要比较的图片
        right_image = process_image(image_2)

        conv_list = []
        for i in range(2):
            output_distance = sess.run([distance], feed_dict={left: left_image, right: right_image, is_training:None})
            #_out_image1 = sess.run([_out], feed_dict={left:right_image})
            #_out_image2 = sess.run([_out], feed_dict={left: left_image})
            #print(_out_image1[0])
            #print(_out_image1[0])
            print(output_distance[0])
            #conv_list.append(concat_out)

            if output_distance[0] > 0.5:
                print("Yes")
            else:
                print("No")
        # tmp =conv_list[0]-conv_list[1]
        # print(np.asarray(conv_list[0]-conv_list[1]).flatten())