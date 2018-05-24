import cv2
import tensorflow as tf
from dataset import *
from resize_image import resize_image

image_1 = cv2.imread('01.jpg')
image_2 = cv2.imread('02.jpg')

def process_image(image1, image2):
    image1 = resize_image(image1)
    image2 = resize_image(image2)
    image1 = cv2.cvtColor(image1,cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(image2,cv2.COLOR_BGR2RGB)

    left_image = np.asarray(np.asarray(image1),dtype='float32')/255.
    right_image = np.asarray(np.asarray(image2),dtype='float32')/255.
    left_image.resize((1, 72, 72, 3))
    right_image.resize((1, 72, 72, 3))

    return left_image, right_image

graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        saver = tf.train.import_meta_graph('checkpoint/model_12000.ckpt.meta')
        saver.restore(sess, 'checkpoint/model_12000.ckpt')
        # 分别取出两个输出层
        left = graph.get_operation_by_name("in/left").outputs[0]
        right = graph.get_operation_by_name("in/right").outputs[0]
        distance = graph.get_operation_by_name("output/distance").outputs[0]

        # 加循环判定，进行多张判定
        left_image, right_image = process_image(image_1, image_2)

        output_distance = sess.run([distance], feed_dict={left: left_image, right: right_image})
        if output_distance[0] > 0.5:
            print("Yes")
        else:
            print("No")