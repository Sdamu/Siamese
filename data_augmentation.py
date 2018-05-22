# 进行数据增强，主要方法是对称，patches 等操作
import os
import cv2
from keras.preprocessing.image import *

AUGMETATION_NUM = 9 # 每张图像扩充的数量

Datagen = ImageDataGenerator(rotation_range=40,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             rescale=None,
                             zoom_range=0.2,
                             fill_mode='nearest',
                             cval=0,
                             horizontal_flip=True
                             )

count = 1

raw_data_path = './AugDataset'
classes_list = os.listdir(raw_data_path)
for class_name in classes_list:
    sub_classes_path = os.path.join(raw_data_path, class_name)
    sub_classes_list = os.listdir(sub_classes_path)
    for sub_class_name in sub_classes_list:
        sub_class_flie_path = os.path.join(sub_classes_path,sub_class_name)
        sub_class_flie_names = os.listdir(sub_class_flie_path)
        for concrete_img_name in sub_class_flie_names:
            this_path = os.path.join(sub_class_flie_path,concrete_img_name)
            img = cv2.imread(this_path)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)   # BGR 2 RGB
            x_raw = img_to_array(img)  # 将图像转换为 np 数组
            x = x_raw.reshape((1,) + x_raw.shape)  # 将图像变为 4 维张量
            index_time = 0
            for batch in Datagen.flow(x, batch_size=1,
                                      save_to_dir=sub_class_flie_path,
                                      save_prefix=class_name + '_' +sub_class_name+ '_'+concrete_img_name[0:2],
                                      save_format='jpeg'):
                index_time += 1
                if index_time > AUGMETATION_NUM-1:
                    break
            print('finished:%f%%' % (count / 10))
            count += 1














# img = cv2.imread('01.jpg') # 加载一个图像
#
# x = img_to_array(img) # 将图像转换为 np 数组
# # print(type(x))
# x = x.reshape((1,) + x.shape)
# print(x.shape)
# i = 0
# #
# for batch in Datagen.flow(x, batch_size=1,
#                               save_to_dir='TEMP', save_prefix='book', save_format='jpeg'):
#     print(i)
#     i += 1
#     if i > 10:
#         break