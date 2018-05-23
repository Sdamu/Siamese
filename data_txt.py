import os
import random
from itertools import combinations

BASEPATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASEPATH, 'AugDataSet')
# 5 个大的分类
class_names = os.listdir(DATA_PATH)

# # 产生正样本对
# ff = open("positive_pairs_path.txt",'wb')
# for class_name in class_names:
#     # 每一大类下的 5 个细分类
#     concrete_class_path = os.path.join(DATA_PATH,class_name)
#     concrete_class_names = os.listdir(concrete_class_path)
#     for concrete_class_name in concrete_class_names:
#         concrete_file_path = os.path.join(concrete_class_path,concrete_class_name)
#         concrete_file_names = os.listdir(concrete_file_path)
#         # 生成从 200 个文件中随机选出两个的全部情况，即 C(20)2
#         combins = [c for c in combinations(range(200), 2)]  # 生成 C(20)2 中情况
#         for combin in combins:
#             (pair_a, pair_b) = combin
#             #print(len(concrete_file_names),pair_a,pair_b)
#             first = os.path.join(concrete_file_path,concrete_file_names[pair_a])
#             second = os.path.join(concrete_file_path,concrete_file_names[pair_b])
#             lines = first + ' '+ second+ '\n'
#             line = lines.encode(encoding="utf-8")
#             # print(line)
#             ff.write(line)
#
# ff.close()

# 生成负样本对
file = open("negative_pairs_path.txt",'wb')
objectPath = []
objectDiffPath = []

for f in os.listdir(DATA_PATH):
    objectPath.append(DATA_PATH + "\\" + f)

numOfClass = len(objectPath)
for i in range(numOfClass):
    object_path = objectPath[i]
    for ff in os.listdir(object_path):
        objectDiffPath.append(object_path + "\\" + ff)

objectCombinations = []
objectCombinations = list(combinations(objectDiffPath, 2))

test = []

for i in range(len(objectCombinations)):
    left = objectCombinations[i][0]
    right = objectCombinations[i][1]
    object_path_path1 = []
    object_path_path2 = []
    for fff in os.listdir(left):
        object_path_path1.append(left + "\\" + fff)
    for ffff in os.listdir(right):
        object_path_path2.append(right + "\\" + ffff)

    for m in range(len(object_path_path1)):
        for n in range(len(object_path_path2)):
            test.append([object_path_path1[m], object_path_path2[n]])


# 在 [0,49000000) 之间随机生成 995000 个随机数，
# 讲这些随机数所对应的行写入 txt 文件

b_list = range(0,49000000)
choose_Id = random.sample(b_list, 995000)

for i in range(len(choose_Id)):
        # 如果 i 在当前选中的索引中，进行写入
        lines = test[choose_Id[i]][0] + ' ' + test[choose_Id[i]][1] + '\n'
        line = lines.encode(encoding="utf-8")
        file.write(line)
        print(choose_Id[i])

file.close()

#
# n = open("positive_pairs_path.txt",'r')
# readlines = n.readlines()
# for line in readlines:
#     c,d = line.split(' ')
#     d = d[:-1]
#     p = cv2.imread(d)
#     cv2.imshow("test",p)
#     cv2.waitKey()
#     pause = 0


