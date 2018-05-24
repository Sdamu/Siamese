import numpy as np
from PIL import Image
import os

# BASE_PATH = 'same_size/lfw/'
# BASE_PATH = '/data/data/faces/small/'
DEV_NUMBER = -10000
batch_size = 128

positive_pairs_path_file = open('positive_pairs_path.txt', 'r')
positive_pairs_path_lines = positive_pairs_path_file.readlines()
negative_pairs_path_file = open('negative_pairs_path.txt', 'r')
negative_pairs_path_lines = negative_pairs_path_file.readlines()


left_image_path_list = []
right_image_path_list = []
similar_list = []

for line in negative_pairs_path_lines:
    left_right = line.strip().split(' ')
    left_image_path_list.append(left_right[0])
    right_image_path_list.append(left_right[1])
    similar_list.append(0)

# negative_left_right = []
# for line in negative_pairs_path_lines:
#     negative_left_right.append(line.strip().split(' '))
#
# negative_pairs_path_lines =[]
#
# np.random.shuffle(negative_left_right)
# for i in range(len(negative_left_right)):
#     if i < len(positive_pairs_path_lines):
#         left_image_path_list.append(negative_left_right[i][0])
#         right_image_path_list.append(negative_left_right[i][1])
#         similar_list.append(0)

for line in positive_pairs_path_lines:
    left_right = line.strip().split(' ')
    left_image_path_list.append(left_right[0])
    right_image_path_list.append(left_right[1])
    similar_list.append(1)

left_image_path_list = np.asarray(left_image_path_list)
right_image_path_list = np.asarray(right_image_path_list)
similar_list = np.asarray(similar_list)

# Randomly shuffle data
np.random.seed(10)
# 生成0~len(similar_list)-1 的数字，最为随机索引
shuffle_indices = np.random.permutation(np.arange(len(similar_list)))
left_shuffled = left_image_path_list[shuffle_indices]
right_shuffled = right_image_path_list[shuffle_indices]
similar_shuffled = similar_list[shuffle_indices]

# print(left_shuffled[:5])
# print(right_shuffled[:5])
# print(similar_shuffled[:5])

# Split train/test set
left_train, left_dev = left_shuffled[:DEV_NUMBER], left_shuffled[DEV_NUMBER:]
right_train, right_dev = right_shuffled[:DEV_NUMBER], right_shuffled[DEV_NUMBER:]
similar_train, similar_dev = similar_shuffled[:DEV_NUMBER], similar_shuffled[DEV_NUMBER:]


# print(left_train[:5])
# print(right_train[:5])
# print(similar_train[:5])


def vectorize_imgs(img_path_list):
    image_arr_list = []
    for img_path in img_path_list:
        if os.path.exists(img_path):
            img = Image.open(img_path)
            img_arr = np.asarray(img, dtype='float32')
            image_arr_list.append(img_arr)
        else:
            print(img_path)
    return image_arr_list


def get_batch_image_path(left_train, right_train, similar_train, start):
    end = (start + batch_size) % len(similar_train)
    if start < end:
        return left_train[start:end], right_train[start:end], similar_train[start:end], end
    # 当 start > end 时，从头返回
    else:
        return np.concatenate([left_train[start:], left_train[:end]]), \
            np.concatenate([right_train[start:], right_train[:end]]), \
            np.concatenate([similar_train[start:], similar_train[:end]]), \
            end


def get_batch_image_array(batch_left, batch_right, batch_similar):
    return np.asarray(vectorize_imgs(batch_left), dtype='float32') / 255., \
           np.asarray(vectorize_imgs(batch_right), dtype='float32') / 255., \
           np.asarray(batch_similar)[:, np.newaxis]


if __name__ == '__main__':
    pass
    # idx = 0
    # batch_left, batch_right, batch_similar, idx = get_batch_image_path(left_train, right_train, similar_train, idx)
    # print(batch_left[:5])
    # print(batch_right[:5])
    # print(batch_similar[:5])
    # print(get_batch_image_array(batch_left[:5], batch_right[:5], batch_similar[:5]))
